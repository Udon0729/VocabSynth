"""トークン固有 z 正規化による文脈弁別能力の評価。

各構成サブワードの logit を、そのトークン固有の平均・標準偏差で
z 正規化してから集約する。汎用サブワード（N, C, P 等）が持つ
文脈非依存の高 logit バイアスを、トークンごとの変動幅で正規化し、
文脈依存の「驚き」信号のみを抽出する。

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_z_normalized.py
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
    BASELINE_PROMPTS,
    AggregationMethod,
    build_component_logit_head,
    compute_logit_statistics,
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
    output_dir = Path("results/z_normalized")
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

    # --- ベースライン統計量の事前計算 ---
    print(
        f"\nベースライン統計量推定中（{len(BASELINE_PROMPTS)} プロンプト）...",
        flush=True,
    )
    bl_mean, bl_std = compute_logit_statistics(model, tokenizer, BASELINE_PROMPTS)
    print(
        f"  平均の平均={bl_mean.mean():.3f}  "
        f"標準偏差の平均={bl_std.mean():.3f}  "
        f"標準偏差の最小={bl_std.min():.3f}",
        flush=True,
    )

    # --- 構成サブワードの統計量を表示 ---
    print("\n=== 構成サブワードのベースライン統計量 ===", flush=True)
    print(
        f"  {'仮想語彙':18s} {'構成語':10s}   "
        f"{'サブワード':>8s}  {'平均':>6s}  {'標準偏差':>6s}  {'z閾値':>6s}",
        flush=True,
    )
    for vtoken in registry:
        for word in vtoken.components:
            ids = tokenizer.encode(word, add_special_tokens=False)
            parts = []
            for tid in ids:
                tok_str = tokenizer.decode([tid])
                mu = bl_mean[tid].item()
                sigma = bl_std[tid].item()
                parts.append(f"{tok_str}={mu:.2f}±{sigma:.2f}")
            print(
                f"  {vtoken.surface:18s} {word:10s} → {' '.join(parts)}",
                flush=True,
            )

    # --- 評価条件の構築 ---
    logit_heads = {}

    # 生 logit（ベースライン）
    for agg in AggregationMethod:
        logit_heads[f"raw/{agg.value}"] = build_component_logit_head(
            tokenizer, registry, agg,
        )

    # 平均差し引き
    for agg in AggregationMethod:
        logit_heads[f"mean_sub/{agg.value}"] = build_component_logit_head(
            tokenizer, registry, agg, baseline=bl_mean,
        )

    # z 正規化
    for agg in AggregationMethod:
        logit_heads[f"z_norm/{agg.value}"] = build_component_logit_head(
            tokenizer, registry, agg,
            baseline=bl_mean, baseline_std=bl_std,
        )

    print(f"\n評価条件: {list(logit_heads.keys())}", flush=True)
    print("\n=== 評価実行 ===", flush=True)
    all_results = evaluate_all(
        model, tokenizer, logit_heads, registry, OUTPUT_PROMPTS,
    )

    # ========== 全条件比較 ==========
    print("\n=== 全条件比較 ===", flush=True)
    print(
        f"{'条件':30s} | {'仮想内正解':>10s} | "
        f"{'対象スコア':>9s} | {'非対象スコア':>10s}",
        flush=True,
    )
    print("-" * 75, flush=True)

    for cond_name in logit_heads:
        correct, total, _ = compute_intra_virtual_accuracy(
            all_results, cond_name,
        )
        tr = [r for r in all_results
              if r["condition"] == cond_name and r["is_target"]]
        nt = [r for r in all_results
              if r["condition"] == cond_name and not r["is_target"]]

        avg_logit = sum(r["logit"] for r in tr) / max(len(tr), 1)
        avg_nt_logit = sum(r["logit"] for r in nt) / max(len(nt), 1)

        print(
            f"  {cond_name:28s} | {correct:3d}/{total:2d}"
            f" ({100 * correct / max(total, 1):4.0f}%)"
            f" | {avg_logit:9.3f} | {avg_nt_logit:10.3f}",
            flush=True,
        )

    # ========== 最良 z_norm 条件の詳細 ==========
    z_conds = [k for k in logit_heads if k.startswith("z_norm/")]
    z_accs = {}
    for c in z_conds:
        cc, tt, _ = compute_intra_virtual_accuracy(all_results, c)
        z_accs[c] = cc
    best_z = max(z_accs, key=z_accs.get)

    print(f"\n=== {best_z}: 仮想語彙内詳細 ===", flush=True)
    _, _, details = compute_intra_virtual_accuracy(all_results, best_z)
    for d in details:
        mark = "○" if d["correct"] else "×"
        print(
            f"  {mark} 対象={d['target']:18s} 仮想内順位={d['intra_rank']}/5"
            f" | {d['prompt'][:55]}",
            flush=True,
        )

    # ========== 代表プロンプトでの全仮想語彙スコア ==========
    print(
        f"\n=== {best_z}: 代表プロンプトでの全仮想語彙スコア ===",
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
            if r["condition"] == best_z
            and r["target_word"] == target
            and r["prompt"] == prompt
        ]
        for e in sorted(entries, key=lambda x: -x["logit"]):
            mark = "→" if e["ranked_word"] == target else " "
            print(
                f"    {mark} {e['ranked_word']:18s} z={e['logit']:7.3f}"
                f" {'(対象)' if e['is_target'] else ''}",
                flush=True,
            )

    # ========== 生 logit との比較: 同一プロンプトの生 vs z ==========
    best_raw = "raw/word_min"
    print(
        f"\n=== 生 logit vs z 正規化: 代表プロンプト比較 ===",
        flush=True,
    )
    for target, prompt in test_prompts:
        print(f'\n  プロンプト: "{prompt}"', flush=True)
        for cond_label, cond_name in [("生", best_raw), ("z", best_z)]:
            entries = [
                r for r in all_results
                if r["condition"] == cond_name
                and r["target_word"] == target
                and r["prompt"] == prompt
            ]
            sorted_e = sorted(entries, key=lambda x: -x["logit"])
            top_name = sorted_e[0]["ranked_word"]
            target_entry = next(e for e in sorted_e if e["ranked_word"] == target)
            target_rank = next(
                i for i, e in enumerate(sorted_e) if e["ranked_word"] == target
            )
            print(
                f"    {cond_label}: 1位={top_name:18s}"
                f" 対象={target:18s} 仮想内順位={target_rank}/5"
                f" スコア={target_entry['logit']:7.3f}",
                flush=True,
            )

    # ========== 保存 ==========
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "timestamp": timestamp,
            "conditions": list(logit_heads.keys()),
            "baseline_stats": {
                "mean_of_means": bl_mean.mean().item(),
                "mean_of_stds": bl_std.mean().item(),
                "min_std": bl_std.min().item(),
            },
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)


if __name__ == "__main__":
    main()
