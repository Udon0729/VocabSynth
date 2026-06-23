"""OMP（直交マッチング追跡）による疎線形結合合成の評価。

平均合成（ベースライン）、z 正規化（現行最良の静的手法、67%）と
OMP 合成（sparsity=16, 32, 64）を比較し、訓練不要の純粋な静的合成
の上限を確認する。

OMP の要点:
  構成サブワードの出力重みの平均をターゲットとし、出力重み行列を
  辞書として疎な線形結合で近似する。構成サブワード自体は辞書から
  除外し、非自明な解を強制する。合成された 1 本の出力重みベクトルを
  VirtualLogitHead 方式（h @ w_z）で logit 計算に使う。

比較条件:
  1. raw/word_mean（構成トークン logit 集約、ベースライン）
  2. z_norm/word_mean（z 正規化、現行最良）
  3. omp_sp16（OMP 合成、sparsity=16、構成サブワード除外）
  4. omp_sp32（OMP 合成、sparsity=32、構成サブワード除外）
  5. omp_sp64（OMP 合成、sparsity=64、構成サブワード除外）

実行例::

    CUDA_VISIBLE_DEVICES=6 nohup uv run python scripts/eval_omp.py > logs/eval_omp.log 2>&1 &
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
from vocabsynth.omp_composer import (
    build_omp_virtual_logit_head,
    build_omp_component_logit_head,
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

SPARSITY_VALUES = [16, 32, 64]


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


def print_omp_diagnostics(diagnostics, label):
    """OMP 合成の診断情報を整形出力する。"""
    print(f"\n=== OMP 診断: {label} ===", flush=True)
    for surface, diag in diagnostics.items():
        if "words" in diag:
            # 語レベル合成の診断
            print(f"  {surface}:", flush=True)
            for wd in diag["words"]:
                print(
                    f"    {wd['word']}:"
                    f"  残差={wd['residual_norm']:.4f}"
                    f"  余弦類似度={wd['cosine_similarity']:.4f}",
                    flush=True,
                )
                for idx, tok, coeff in wd["top_atoms"][:3]:
                    print(
                        f"      原子: '{tok}' (id={idx})"
                        f"  係数={coeff:.4f}",
                        flush=True,
                    )
        else:
            # 仮想語彙レベル合成の診断
            print(
                f"  {surface}:"
                f"  残差={diag['residual_norm']:.4f}"
                f"  余弦類似度={diag['cosine_similarity']:.4f}"
                f"  ターゲットノルム={diag['target_norm']:.4f}"
                f"  合成ノルム={diag['composed_norm']:.4f}"
                f"  選択原子数={diag['num_selected']}",
                flush=True,
            )
            print("    上位原子:", flush=True)
            for idx, tok, coeff in diag["top_atoms"][:5]:
                print(
                    f"      '{tok}' (id={idx}) 係数={coeff:.4f}",
                    flush=True,
                )


def main() -> None:
    output_dir = Path("results/omp")
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

    # --- 出力重み行列の基本統計 ---
    dict_norms = W.norm(dim=1)
    print(
        f"\n出力重み行列: 形状={list(W.shape)}"
        f"  ノルム: 平均={dict_norms.mean():.4f}"
        f"  中央値={dict_norms.median():.4f}"
        f"  標準偏差={dict_norms.std():.4f}",
        flush=True,
    )

    # --- 構成の確認 ---
    print("\n=== 語レベル構成の確認 ===", flush=True)
    for vtoken in registry:
        word_parts = []
        all_ids = []
        for word in vtoken.components:
            ids = tokenizer.encode(word, add_special_tokens=False)
            all_ids.extend(ids)
            tokens_str = [tokenizer.decode([t]) for t in ids]
            word_parts.append(f"{word} -> {tokens_str}")
        avg_norm = W[all_ids].mean(dim=0).norm().item()
        print(
            f"  {vtoken.surface:18s}: {' + '.join(word_parts)}"
            f"  (平均ノルム={avg_norm:.4f})",
            flush=True,
        )

    # --- ベースライン統計量の事前計算 ---
    print(
        f"\nベースライン統計量推定中（{len(BASELINE_PROMPTS)} プロンプト）...",
        flush=True,
    )
    bl_mean, bl_std = compute_logit_statistics(model, tokenizer, BASELINE_PROMPTS)
    print(
        f"  平均の平均={bl_mean.mean():.3f}"
        f"  標準偏差の平均={bl_std.mean():.3f}"
        f"  標準偏差の最小={bl_std.min():.3f}",
        flush=True,
    )

    # --- 評価条件の構築 ---
    logit_heads = {}

    # 条件1: raw/word_mean（構成トークン logit 集約、ベースライン）
    logit_heads["raw/word_mean"] = build_component_logit_head(
        tokenizer, registry, AggregationMethod.WORD_MEAN,
    )

    # 条件2: z_norm/word_mean（z 正規化、現行最良の静的手法）
    logit_heads["z_norm/word_mean"] = build_component_logit_head(
        tokenizer, registry, AggregationMethod.WORD_MEAN,
        baseline=bl_mean, baseline_std=bl_std,
    )

    # 条件3-5: OMP 合成（仮想語彙レベル、構成サブワード除外）
    W_cpu = W.cpu().float()
    all_diagnostics = {}
    for sp in SPARSITY_VALUES:
        print(f"\nOMP ヘッド構築中（sparsity={sp}、構成サブワード除外）...", flush=True)
        head, diag = build_omp_virtual_logit_head(
            tokenizer, registry, W_cpu,
            sparsity=sp,
            norm_correction=True,
            exclude_components=True,
        )
        cond_name = f"omp_sp{sp}"
        # OMP は CPU 上で合成するため、出力重みを GPU に移す
        head._U = head._U.to(device)
        logit_heads[cond_name] = head
        all_diagnostics[cond_name] = diag
        print_omp_diagnostics(diag, cond_name)
        print(f"  構築完了", flush=True)

    # 追加条件: 語レベル OMP + word_mean（参考）
    print(f"\n語レベル OMP ヘッド構築中（sparsity=32、構成サブワード除外）...", flush=True)
    head_wm, diag_wm = build_omp_component_logit_head(
        tokenizer, registry, W_cpu,
        sparsity=32,
        norm_correction=True,
        exclude_components=True,
    )
    head_wm._U = head_wm._U.to(device)
    logit_heads["omp_word_sp32"] = head_wm
    all_diagnostics["omp_word_sp32"] = diag_wm
    print_omp_diagnostics(diag_wm, "omp_word_sp32")
    print(f"  構築完了", flush=True)

    print(f"\n評価条件: {list(logit_heads.keys())}", flush=True)

    # --- 評価実行 ---
    print("\n=== 評価実行 ===", flush=True)
    all_results = evaluate_all(
        model, tokenizer, logit_heads, registry, OUTPUT_PROMPTS,
    )

    # ========== 全条件比較 ==========
    print("\n=== 全条件比較 ===", flush=True)
    print(
        f"{'条件':30s} | {'仮想内正解':>10s} | "
        f"{'対象スコア':>9s} | {'非対象スコア':>10s} | {'差分':>8s}",
        flush=True,
    )
    print("-" * 85, flush=True)

    summary = {}
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
        margin = avg_logit - avg_nt_logit

        acc = correct / max(total, 1)
        summary[cond_name] = {
            "correct": correct,
            "total": total,
            "accuracy": acc,
            "avg_target_logit": avg_logit,
            "avg_nontarget_logit": avg_nt_logit,
            "margin": margin,
        }

        print(
            f"  {cond_name:28s} | {correct:3d}/{total:2d}"
            f" ({100 * acc:4.0f}%)"
            f" | {avg_logit:9.3f} | {avg_nt_logit:10.3f}"
            f" | {margin:8.3f}",
            flush=True,
        )

    # ========== 各条件の詳細 ==========
    for cond_name in logit_heads:
        print(f"\n=== {cond_name}: 仮想語彙内詳細 ===", flush=True)
        _, _, details = compute_intra_virtual_accuracy(all_results, cond_name)
        for d in details:
            mark = "○" if d["correct"] else "×"
            print(
                f"  {mark} 対象={d['target']:18s}"
                f" 仮想内順位={d['intra_rank']}/5"
                f" | {d['prompt'][:55]}",
                flush=True,
            )

    # ========== 代表プロンプトでの全仮想語彙スコア ==========
    print("\n=== 代表プロンプトでの全仮想語彙スコア ===", flush=True)
    test_prompts = [
        ("NaritaCake", "In Narita, a popular Cake is"),
        ("BerlinPretzel", "In Berlin, a popular Pretzel is"),
        ("TokyoBridge", "In Tokyo, a well-known Bridge is"),
    ]
    for target, prompt in test_prompts:
        print(f'\n  プロンプト: "{prompt}"', flush=True)
        for cond_name in logit_heads:
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
            is_correct = "○" if target_rank == 0 else "×"
            print(
                f"    {is_correct} {cond_name:28s}"
                f"  1位={top_name:18s}"
                f"  対象順位={target_rank}/5"
                f"  スコア={target_entry['logit']:7.3f}",
                flush=True,
            )

    # ========== まとめ ==========
    print("\n=== まとめ ===", flush=True)
    baseline_acc = summary.get("z_norm/word_mean", {}).get("accuracy", 0)
    raw_acc = summary.get("raw/word_mean", {}).get("accuracy", 0)
    print(
        f"  生 logit 集約（raw/word_mean）: {raw_acc:.0%}",
        flush=True,
    )
    print(
        f"  現行最良（z_norm/word_mean）: {baseline_acc:.0%}",
        flush=True,
    )
    for sp in SPARSITY_VALUES:
        key = f"omp_sp{sp}"
        if key in summary:
            omp_acc = summary[key]["accuracy"]
            diff = omp_acc - baseline_acc
            direction = "改善" if diff > 0 else ("同等" if diff == 0 else "劣化")
            print(
                f"  OMP 仮想語彙レベル (sparsity={sp:3d}): {omp_acc:.0%}"
                f"  （{direction}: {diff:+.0%}）",
                flush=True,
            )

    wm_key = "omp_word_sp32"
    if wm_key in summary:
        wm_acc = summary[wm_key]["accuracy"]
        diff = wm_acc - baseline_acc
        direction = "改善" if diff > 0 else ("同等" if diff == 0 else "劣化")
        print(
            f"  OMP 語レベル+word_mean (sparsity=32): {wm_acc:.0%}"
            f"  （{direction}: {diff:+.0%}）",
            flush=True,
        )

    # ========== 結果保存 ==========
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"

    # 診断情報をシリアライズ可能に変換
    serializable_diag = {}
    for cond_name, diag in all_diagnostics.items():
        serializable_diag[cond_name] = {}
        for surface, d in diag.items():
            if "words" in d:
                serializable_diag[cond_name][surface] = {
                    "words": [
                        {
                            "word": wd["word"],
                            "residual_norm": wd["residual_norm"],
                            "cosine_similarity": wd["cosine_similarity"],
                            "top_atoms": [
                                {"id": idx, "token": tok, "coeff": coeff}
                                for idx, tok, coeff in wd["top_atoms"]
                            ],
                        }
                        for wd in d["words"]
                    ],
                }
            else:
                serializable_diag[cond_name][surface] = {
                    "target_norm": d["target_norm"],
                    "composed_norm": d["composed_norm"],
                    "residual_norm": d["residual_norm"],
                    "cosine_similarity": d["cosine_similarity"],
                    "num_selected": d["num_selected"],
                    "top_atoms": [
                        {"id": idx, "token": tok, "coeff": coeff}
                        for idx, tok, coeff in d["top_atoms"]
                    ],
                }

    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "timestamp": timestamp,
            "conditions": list(logit_heads.keys()),
            "sparsity_values": SPARSITY_VALUES,
            "summary": {
                k: {
                    "correct": v["correct"],
                    "total": v["total"],
                    "accuracy": v["accuracy"],
                    "avg_target_logit": v["avg_target_logit"],
                    "avg_nontarget_logit": v["avg_nontarget_logit"],
                    "margin": v["margin"],
                }
                for k, v in summary.items()
            },
            "diagnostics": serializable_diag,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)
    print("完了", flush=True)


if __name__ == "__main__":
    main()
