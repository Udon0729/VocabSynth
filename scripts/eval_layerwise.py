"""中間層 logit lens による文脈弁別能力の層別評価。

各層の隠れ状態を最終層正規化＋出力重み行列で射影し（logit lens）、
構成トークン logit を読み出して仮想語彙内正解率を層ごとに比較する。

最終層は次トークン予測に特化しているため既出語の logit が抑制されるが、
中間層ではこの抑制が弱く、文脈の主題的情報が残る可能性がある。

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_layerwise.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vocabsynth.analyzer import TokenizerAnalyzer
from vocabsynth.logit_head import (
    AggregationMethod,
    ComponentLogitHead,
    LayerAggregation,
    LayerwiseLogitHead,
    build_component_logit_head,
    build_layerwise_logit_head,
)
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
    """全条件を一度の順伝播で評価する。"""
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
                if isinstance(head, LayerwiseLogitHead):
                    rankings = head.get_virtual_rankings(
                        out.hidden_states, out.logits,
                    )
                else:
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
    output_dir = Path("results/layerwise")
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

    W = model.get_output_embeddings().weight.detach()
    final_norm = model.gpt_neox.final_layer_norm
    num_layers = model.config.num_hidden_layers
    print(f"層数: {num_layers}（hidden_states は {num_layers + 1} 要素）", flush=True)

    logit_heads = {}

    # --- ベースライン: 最終層の通常 component/word_min ---
    logit_heads["baseline/word_min"] = build_component_logit_head(
        tokenizer, registry, AggregationMethod.WORD_MIN,
    )

    # --- 各層の logit lens（WORD_MIN 集約）---
    for layer_idx in range(num_layers + 1):
        head = build_layerwise_logit_head(
            tokenizer, registry, W, final_norm,
            AggregationMethod.WORD_MIN,
            layer_indices=[layer_idx],
            layer_aggregation=LayerAggregation.SINGLE,
        )
        logit_heads[f"lens/L{layer_idx:02d}"] = head

    # --- 層範囲の平均 ---
    layer_ranges = {
        "early(0-7)": list(range(8)),
        "middle(8-15)": list(range(8, 16)),
        "late(16-23)": list(range(16, 24)),
        "mid-late(12-23)": list(range(12, 24)),
        "all(0-24)": list(range(num_layers + 1)),
    }
    for name, indices in layer_ranges.items():
        head = build_layerwise_logit_head(
            tokenizer, registry, W, final_norm,
            AggregationMethod.WORD_MIN,
            layer_indices=indices,
            layer_aggregation=LayerAggregation.MEAN,
        )
        logit_heads[f"mean/{name}"] = head

    # --- 全層の max ---
    logit_heads["max/all(0-24)"] = build_layerwise_logit_head(
        tokenizer, registry, W, final_norm,
        AggregationMethod.WORD_MIN,
        layer_indices=list(range(num_layers + 1)),
        layer_aggregation=LayerAggregation.MAX,
    )

    print(f"\n評価条件数: {len(logit_heads)}", flush=True)
    print("\n=== 評価実行 ===", flush=True)
    all_results = evaluate_all(
        model, tokenizer, logit_heads, registry, OUTPUT_PROMPTS,
    )

    # ========== 層別正解率テーブル ==========
    print("\n=== 層別 logit lens 正解率（WORD_MIN 集約）===", flush=True)
    print(
        f"{'層':>5s} | {'正解率':>10s} | {'順位中央値':>8s} "
        f"| {'対象logit':>9s} | {'選択性比':>8s}",
        flush=True,
    )
    print("-" * 65, flush=True)

    layer_accs = {}
    for layer_idx in range(num_layers + 1):
        cond = f"lens/L{layer_idx:02d}"
        correct, total, _ = compute_intra_virtual_accuracy(all_results, cond)
        layer_accs[layer_idx] = (correct, total)

        tr = [r for r in all_results
              if r["condition"] == cond and r["is_target"]]
        nt = [r for r in all_results
              if r["condition"] == cond and not r["is_target"]]

        med_rank = sorted(r["rank"] for r in tr)[len(tr) // 2]
        avg_logit = sum(r["logit"] for r in tr) / max(len(tr), 1)
        avg_rank = sum(r["rank"] for r in tr) / max(len(tr), 1)
        avg_nt_rank = sum(r["rank"] for r in nt) / max(len(nt), 1)
        ratio = avg_nt_rank / max(avg_rank, 1)

        bar = "█" * correct + "░" * (total - correct)
        print(
            f"  L{layer_idx:02d} | {correct:2d}/{total:2d} ({100 * correct / total:3.0f}%)"
            f" | {med_rank:8d} | {avg_logit:9.3f} | {ratio:7.1f}x  {bar}",
            flush=True,
        )

    # ベースライン行
    cond = "baseline/word_min"
    correct, total, _ = compute_intra_virtual_accuracy(all_results, cond)
    tr = [r for r in all_results
          if r["condition"] == cond and r["is_target"]]
    nt = [r for r in all_results
          if r["condition"] == cond and not r["is_target"]]
    med_rank = sorted(r["rank"] for r in tr)[len(tr) // 2]
    avg_logit = sum(r["logit"] for r in tr) / max(len(tr), 1)
    avg_rank = sum(r["rank"] for r in tr) / max(len(tr), 1)
    avg_nt_rank = sum(r["rank"] for r in nt) / max(len(nt), 1)
    ratio = avg_nt_rank / max(avg_rank, 1)
    bar = "█" * correct + "░" * (total - correct)
    print(
        f"  最終 | {correct:2d}/{total:2d} ({100 * correct / total:3.0f}%)"
        f" | {med_rank:8d} | {avg_logit:9.3f} | {ratio:7.1f}x  {bar}"
        "  ← component/word_min",
        flush=True,
    )

    # ========== 層範囲集約 ==========
    print("\n=== 層範囲集約 ===", flush=True)
    for cond_name in logit_heads:
        if not (cond_name.startswith("mean/") or cond_name.startswith("max/")):
            continue
        correct, total, _ = compute_intra_virtual_accuracy(
            all_results, cond_name,
        )
        tr = [r for r in all_results
              if r["condition"] == cond_name and r["is_target"]]
        med_rank = sorted(r["rank"] for r in tr)[len(tr) // 2]
        avg_logit = sum(r["logit"] for r in tr) / max(len(tr), 1)
        print(
            f"  {cond_name:25s} | {correct:2d}/{total:2d}"
            f" ({100 * correct / total:3.0f}%) | 中央値={med_rank:5d}"
            f" | logit={avg_logit:7.3f}",
            flush=True,
        )

    # ========== 最良層の詳細 ==========
    best_L = max(layer_accs, key=lambda k: layer_accs[k][0])
    best_cond = f"lens/L{best_L:02d}"
    best_c, best_t, best_details = compute_intra_virtual_accuracy(
        all_results, best_cond,
    )
    print(
        f"\n=== 最良層 L{best_L}（{best_c}/{best_t}）の仮想語彙内詳細 ===",
        flush=True,
    )
    for d in best_details:
        mark = "○" if d["correct"] else "×"
        print(
            f"  {mark} 対象={d['target']:18s} 仮想内順位={d['intra_rank']}/5"
            f" | {d['prompt'][:55]}",
            flush=True,
        )

    # ========== 最良層の代表プロンプト ==========
    print(
        f"\n=== L{best_L}: 代表プロンプトでの全仮想語彙スコア ===",
        flush=True,
    )
    test_prompts = [
        ("NaritaCake", "In Narita, a popular Cake is"),
        ("BerlinPretzel", "In Berlin, a popular Pretzel is"),
        ("TokyoBridge", "In Tokyo, a well-known Bridge is"),
    ]
    for target, prompt in test_prompts:
        print(f'\n  プロンプト: "{prompt}"', flush=True)
        entries = [
            r for r in all_results
            if r["condition"] == best_cond
            and r["target_word"] == target
            and r["prompt"] == prompt
        ]
        for e in sorted(entries, key=lambda x: -x["logit"]):
            mark = "→" if e["ranked_word"] == target else " "
            print(
                f"    {mark} {e['ranked_word']:18s} logit={e['logit']:7.3f}"
                f" 順位={e['rank']:5d}"
                f" {'(対象)' if e['is_target'] else ''}",
                flush=True,
            )

    # ========== 保存 ==========
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "timestamp": timestamp,
            "num_layers": num_layers,
            "conditions": list(logit_heads.keys()),
            "layer_accuracies": {
                f"L{layer_idx}": {"correct": c, "total": t}
                for layer_idx, (c, t) in layer_accs.items()
            },
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)


if __name__ == "__main__":
    main()
