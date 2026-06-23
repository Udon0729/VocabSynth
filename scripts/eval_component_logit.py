"""構成トークン logit 集約による文脈弁別能力の評価。

静的出力重み合成（直接合成、GW）と構成トークン logit 集約
（FLAT_MEAN、WORD_MEAN、WORD_MIN）を比較し、正しい仮想語彙が
正しい文脈で選ばれるかを検証する。

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_component_logit.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vocabsynth.analyzer import TokenizerAnalyzer
from vocabsynth.composer import ComposeMethod
from vocabsynth.logit_head import (
    BASELINE_PROMPTS,
    AggregationMethod,
    ComponentLogitHead,
    GateMethod,
    HiddenStateGate,
    VirtualLogitHead,
    build_component_logit_head,
    build_hidden_state_gate,
    compute_logit_baseline,
)
from vocabsynth.ot_composer import OTMethod, OTOutputComposer
from vocabsynth.registry import VocabularyRegistry

MODEL_NAME = "EleutherAI/pythia-410m"

VIRTUAL_TOKENS = [
    {"surface": "NaritaCake", "components": ["Narita", "Cake"], "relation": "place+food"},
    {"surface": "OsakaNoodle", "components": ["Osaka", "Noodle"], "relation": "place+food"},
    {"surface": "BerlinPretzel", "components": ["Berlin", "Pretzel"], "relation": "place+food"},
    {"surface": "ParisChocolate", "components": ["Paris", "Chocolate"], "relation": "place+food"},
    {"surface": "LondonPie", "components": ["London", "Pie"], "relation": "place+food"},
    {"surface": "TokyoBridge", "components": ["Tokyo", "Bridge"], "relation": "place+structure"},
]

OUTPUT_PROMPTS = {
    "place+food": [
        ("A famous local {head} from {modifier} is called", {"modifier": 0, "head": 1}),
        ("The specialty food of {modifier} known as a {head} is", {"modifier": 0, "head": 1}),
        ("In {modifier}, a popular {head} is", {"modifier": 0, "head": 1}),
    ],
    "place+structure": [
        ("A famous {head} in {modifier} is called", {"modifier": 0, "head": 1}),
        ("The iconic {head} of {modifier} is", {"modifier": 0, "head": 1}),
        ("In {modifier}, a well-known {head} is", {"modifier": 0, "head": 1}),
    ],
}


@torch.no_grad()
def evaluate_all(model, tokenizer, logit_heads, registry, prompts_map):
    """全条件×全仮想語彙×全プロンプトの結果を一括で返す。"""
    device = next(model.parameters()).device
    all_results = []

    for vtoken in registry:
        rel_key = vtoken.relation.value
        for tmpl, keys in prompts_map.get(rel_key, []):
            kwargs = {k: vtoken.components[v] for k, v in keys.items()}
            prompt = tmpl.format(**kwargs)

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            out = model(input_ids=input_ids, output_hidden_states=True)

            for cond_name, head in logit_heads.items():
                rankings = head.get_virtual_rankings(
                    out.hidden_states[-1], out.logits,
                )
                for r in rankings:
                    all_results.append({
                        "condition": cond_name,
                        "target_word": vtoken.surface,
                        "ranked_word": r["surface"],
                        "is_target": r["surface"] == vtoken.surface,
                        "prompt": prompt,
                        "rank": r["rank"],
                        "logit": r["logit"],
                    })

    return all_results


def compute_intra_virtual_accuracy(results, condition):
    """仮想語彙内で対象語が最上位になる割合を計算する。"""
    cond_data = [r for r in results if r["condition"] == condition]

    by_prompt = defaultdict(list)
    for r in cond_data:
        by_prompt[(r["target_word"], r["prompt"])].append(r)

    correct = 0
    total = 0
    details = []
    for (target_word, prompt), entries in sorted(by_prompt.items()):
        sorted_e = sorted(entries, key=lambda x: -x["logit"])
        is_correct = sorted_e[0]["ranked_word"] == target_word
        if is_correct:
            correct += 1
        total += 1

        target_intra_rank = next(
            i for i, e in enumerate(sorted_e)
            if e["ranked_word"] == target_word
        )
        details.append({
            "target": target_word,
            "prompt": prompt,
            "intra_rank": target_intra_rank,
            "correct": is_correct,
        })

    return correct, total, details


def main() -> None:
    output_dir = Path("results/component_logit")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"モデル読み込み: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"デバイス: {device}", flush=True)

    registry = VocabularyRegistry()
    registry.add_from_dicts(VIRTUAL_TOKENS)
    analyzer = TokenizerAnalyzer(tokenizer)
    analyzer.analyze_registry(registry)

    E = model.get_input_embeddings().weight.detach()
    W = model.get_output_embeddings().weight.detach()

    # --- ベースライン logit の事前計算 ---
    print(f"\nベースライン logit 推定中（{len(BASELINE_PROMPTS)} プロンプト）...", flush=True)
    baseline = compute_logit_baseline(model, tokenizer, BASELINE_PROMPTS)
    print(f"  ベースライン統計: 平均={baseline.mean():.3f} 標準偏差={baseline.std():.3f}", flush=True)

    # 構成トークンのベースライン値を確認
    print("\n=== 構成サブワードのベースライン logit ===", flush=True)
    for vtoken in registry:
        for word in vtoken.components:
            ids = tokenizer.encode(word, add_special_tokens=False)
            tokens_str = [tokenizer.decode([t]) for t in ids]
            baselines_str = " ".join(
                f"{s}={baseline[t]:.2f}" for s, t in zip(tokens_str, ids)
            )
            print(f"  {vtoken.surface:18s} {word:10s} → {baselines_str}", flush=True)

    logit_heads: dict[str, VirtualLogitHead | ComponentLogitHead] = {}

    # --- 静的重み合成ベースライン ---
    ot_composer = OTOutputComposer(E, W, k_neighbors=200, epsilon=0.05)

    direct_rows = []
    for vtoken in registry:
        w = ot_composer.compose_direct(
            vtoken.component_token_ids, ComposeMethod.MEAN, tokenizer,
        )
        direct_rows.append(w)
    logit_heads["static/direct"] = VirtualLogitHead(
        torch.stack(direct_rows).to(device), registry.surfaces(),
    )

    ot_composer._bary_cache.clear()
    gw_rows = []
    for vtoken in registry:
        w = ot_composer.compose(
            vtoken.component_token_ids, ComposeMethod.MEAN,
            OTMethod.GW, tokenizer,
        )
        gw_rows.append(w)
    logit_heads["static/gw"] = VirtualLogitHead(
        torch.stack(gw_rows).to(device), registry.surfaces(),
    )

    # --- 構成トークン logit 集約（生 logit）---
    for agg in AggregationMethod:
        head = build_component_logit_head(tokenizer, registry, agg)
        logit_heads[f"component/{agg.value}"] = head

    # --- 構成トークン logit 集約（ベースライン差し引き）---
    for agg in AggregationMethod:
        head = build_component_logit_head(
            tokenizer, registry, agg, baseline=baseline,
        )
        logit_heads[f"baseline/{agg.value}"] = head

    # --- 隠れ状態ゲート ---
    for gm in GateMethod:
        head = build_hidden_state_gate(tokenizer, registry, E, gm)
        logit_heads[f"gate/{gm.value}"] = head

    print(f"\n評価条件: {list(logit_heads.keys())}", flush=True)

    print("\n=== 語レベル構成の確認 ===", flush=True)
    for vtoken in registry:
        word_parts = []
        for word in vtoken.components:
            ids = tokenizer.encode(word, add_special_tokens=False)
            tokens_str = [tokenizer.decode([t]) for t in ids]
            word_parts.append(f"{word} → {tokens_str}")
        print(f"  {vtoken.surface:18s}: {' + '.join(word_parts)}", flush=True)

    print("\n=== 評価実行 ===", flush=True)
    all_results = evaluate_all(
        model, tokenizer, logit_heads, registry, OUTPUT_PROMPTS,
    )

    print("\n=== 全条件比較 ===", flush=True)
    print(
        f"{'条件':30s} | {'仮想内正解':>10s} | {'対象順位':>8s} {'中央値':>6s} "
        f"{'対象logit':>9s} | {'非対象logit':>10s} {'選択性比':>8s}",
        flush=True,
    )
    print("-" * 105, flush=True)

    for cond_name in logit_heads:
        correct, total, details = compute_intra_virtual_accuracy(
            all_results, cond_name,
        )

        tr = [r for r in all_results
              if r["condition"] == cond_name and r["is_target"]]
        nt = [r for r in all_results
              if r["condition"] == cond_name and not r["is_target"]]

        avg_rank = sum(r["rank"] for r in tr) / max(len(tr), 1)
        med_rank = sorted(r["rank"] for r in tr)[len(tr) // 2]
        avg_logit = sum(r["logit"] for r in tr) / max(len(tr), 1)

        avg_nt_logit = sum(r["logit"] for r in nt) / max(len(nt), 1)
        avg_nt_rank = sum(r["rank"] for r in nt) / max(len(nt), 1)
        ratio = avg_nt_rank / max(avg_rank, 1)

        print(
            f"  {cond_name:28s} | {correct:3d}/{total:2d} "
            f"({100*correct/max(total,1):4.0f}%) | {avg_rank:8.0f} {med_rank:6d} "
            f"{avg_logit:9.3f} | {avg_nt_logit:10.3f} {ratio:8.1f}x",
            flush=True,
        )

    # 詳細: gate/gate_word_min の仮想語彙内順位
    best_gate = "gate/gate_word_min"
    gate_candidates = [k for k in logit_heads if k.startswith("gate/")]
    if gate_candidates:
        gate_accs = {}
        for gc in gate_candidates:
            c, t, _ = compute_intra_virtual_accuracy(all_results, gc)
            gate_accs[gc] = c
        best_gate = max(gate_accs, key=gate_accs.get)

    print(f"\n=== {best_gate}: 仮想語彙内詳細 ===", flush=True)
    _, _, gm_details = compute_intra_virtual_accuracy(
        all_results, best_gate,
    )
    for d in gm_details:
        mark = "○" if d["correct"] else "×"
        print(
            f"  {mark} 対象={d['target']:18s} 仮想内順位={d['intra_rank']}/5 | "
            f"{d['prompt'][:55]}",
            flush=True,
        )

    # 詳細: best_gate での各仮想語彙のスコア（文脈弁別の検証）
    print(f"\n=== {best_gate}: 代表プロンプトでの全仮想語彙スコア ===", flush=True)
    test_prompts = [
        ("NaritaCake", "In Narita, a popular Cake is"),
        ("BerlinPretzel", "In Berlin, a popular Pretzel is"),
        ("TokyoBridge", "In Tokyo, a well-known Bridge is"),
    ]
    for target, prompt in test_prompts:
        print(f'\n  プロンプト: "{prompt}"', flush=True)
        entries = [
            r for r in all_results
            if r["condition"] == best_gate
            and r["target_word"] == target
            and r["prompt"] == prompt
        ]
        for e in sorted(entries, key=lambda x: -x["logit"]):
            mark = "→" if e["ranked_word"] == target else " "
            print(
                f"    {mark} {e['ranked_word']:18s} logit={e['logit']:7.3f} "
                f"順位={e['rank']:5d} {'(対象)' if e['is_target'] else ''}",
                flush=True,
            )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "timestamp": timestamp,
            "conditions": list(logit_heads.keys()),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)


if __name__ == "__main__":
    main()
