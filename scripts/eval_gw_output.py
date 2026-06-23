"""Gromov-Wasserstein 重心射影による出力重み合成の評価。

交差空間 OT（ベースライン: 直接合成, UOT）と GW 系手法
（GW, 半緩和 GW, 部分 GW）を比較し、出力候補順位の改善と
logit 膨張の程度を検証する。

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_gw_output.py
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
from vocabsynth.logit_head import VirtualLogitHead
from vocabsynth.ot_composer import OTMethod, OTOutputComposer
from vocabsynth.registry import RelationType, VocabularyRegistry

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

K_NEIGHBORS = 200
EPSILON = 0.05


@torch.no_grad()
def evaluate_rankings(
    model, tokenizer, logit_head, registry, prompts_map,
) -> list[dict]:
    """全仮想語彙×全プロンプトの出力候補順位を評価する。"""
    device = next(model.parameters()).device
    results = []

    for vtoken in registry:
        rel_key = vtoken.relation.value
        for tmpl, keys in prompts_map.get(rel_key, []):
            kwargs = {k: vtoken.components[v] for k, v in keys.items()}
            prompt = tmpl.format(**kwargs)

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            out = model(input_ids=input_ids, output_hidden_states=True)

            rankings = logit_head.get_virtual_rankings(
                out.hidden_states[-1], out.logits,
            )

            for r in rankings:
                results.append({
                    "target_word": vtoken.surface,
                    "ranked_word": r["surface"],
                    "is_target": r["surface"] == vtoken.surface,
                    "prompt": prompt,
                    "rank": r["rank"],
                    "logit": r["logit"],
                    "probability": r["probability"],
                })

    return results


def build_condition_list():
    """評価条件の一覧を構築する。"""
    conditions = []

    conditions.append({
        "label": "direct",
        "ot_method": None,
        "tau": None,
        "partial_mass": None,
    })

    for tau in [0.1, 0.5]:
        conditions.append({
            "label": f"uot_τ{tau}",
            "ot_method": OTMethod.UNBALANCED,
            "tau": tau,
            "partial_mass": None,
        })

    conditions.append({
        "label": "gw",
        "ot_method": OTMethod.GW,
        "tau": None,
        "partial_mass": None,
    })

    conditions.append({
        "label": "srgw",
        "ot_method": OTMethod.SEMIRELAXED_GW,
        "tau": None,
        "partial_mass": None,
    })

    for m in [0.25, 0.5, 0.8]:
        conditions.append({
            "label": f"pgw_m{m}",
            "ot_method": OTMethod.PARTIAL_GW,
            "tau": None,
            "partial_mass": m,
        })

    return conditions


def main() -> None:
    output_dir = Path("results/gw_output")
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

    composer = OTOutputComposer(
        E, W, k_neighbors=K_NEIGHBORS, epsilon=EPSILON,
    )

    print("\n=== 近傍幾何診断（構造整合性を含む）===", flush=True)
    for vtoken in registry:
        for tid in vtoken.component_token_ids:
            diag = composer.diagnose_local_geometry(tid)
            tok_str = tokenizer.decode([tid]).strip()
            print(
                f"  {tok_str:12s} (id={tid:5d}) | "
                f"E-W cos={diag['paired_cosine_mean']:.3f} | "
                f"交差: 対角={diag['diagonal_cost_mean']:.4f} "
                f"非対角={diag['off_diagonal_cost_mean']:.4f} | "
                f"距離構造 Spearman={diag['intra_distance_rank_corr']:.3f}",
                flush=True,
            )

    conditions = build_condition_list()
    all_results = []

    for cond in conditions:
        label = f"mean/{cond['label']}"
        print(f"\n{'='*70}", flush=True)
        print(f"条件: {label}", flush=True)
        print(f"{'='*70}", flush=True)

        if cond["tau"] is not None:
            composer._tau = cond["tau"]
        if cond["partial_mass"] is not None:
            composer._partial_mass = cond["partial_mass"]
        composer._bary_cache.clear()

        output_weight_rows = []
        for vtoken in registry:
            if cond["ot_method"] is None:
                w = composer.compose_direct(
                    vtoken.component_token_ids,
                    ComposeMethod.MEAN, tokenizer,
                )
            else:
                w = composer.compose(
                    vtoken.component_token_ids,
                    ComposeMethod.MEAN,
                    cond["ot_method"],
                    tokenizer,
                )
            output_weight_rows.append(w)

        logit_head = VirtualLogitHead(
            torch.stack(output_weight_rows).to(device),
            registry.surfaces(),
        )

        results = evaluate_rankings(
            model, tokenizer, logit_head, registry, OUTPUT_PROMPTS,
        )

        target_results = [r for r in results if r["is_target"]]
        non_target_results = [r for r in results if not r["is_target"]]

        avg_rank = sum(r["rank"] for r in target_results) / max(len(target_results), 1)
        med_rank = sorted(r["rank"] for r in target_results)[len(target_results) // 2]
        avg_logit = sum(r["logit"] for r in target_results) / max(len(target_results), 1)

        print(f"  対象語: 平均順位={avg_rank:.0f} 中央値={med_rank} 平均logit={avg_logit:.3f}", flush=True)

        if non_target_results:
            avg_nt_logit = sum(r["logit"] for r in non_target_results) / len(non_target_results)
            avg_nt_rank = sum(r["rank"] for r in non_target_results) / len(non_target_results)
            ratio = avg_nt_rank / max(avg_rank, 1)
            print(
                f"  非対象: 平均順位={avg_nt_rank:.0f} 平均logit={avg_nt_logit:.3f} "
                f"選択性比={ratio:.1f}x",
                flush=True,
            )

        for vtoken in registry:
            vr = [r for r in target_results if r["target_word"] == vtoken.surface]
            if vr:
                v_avg = sum(r["rank"] for r in vr) / len(vr)
                v_best = min(r["rank"] for r in vr)
                print(f"    {vtoken.surface:18s} 平均={v_avg:.0f} 最良={v_best}", flush=True)

        for r in results:
            r["condition"] = label
            r["ot_method"] = cond["label"]
            r["tau"] = cond["tau"]
            r["partial_mass"] = cond["partial_mass"]
        all_results.extend(results)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "timestamp": timestamp,
            "k_neighbors": K_NEIGHBORS,
            "epsilon": EPSILON,
            "conditions": [c["label"] for c in conditions],
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)

    print("\n\n=== 全条件比較 ===", flush=True)
    print(
        f"{'条件':28s} | {'対象順位':>8s} {'中央値':>6s} {'対象logit':>9s} | "
        f"{'非対象logit':>10s} {'非対象順位':>9s} {'選択性比':>8s}",
        flush=True,
    )
    print("-" * 95, flush=True)

    for cond in conditions:
        label = f"mean/{cond['label']}"
        tr = [r for r in all_results if r["condition"] == label and r["is_target"]]
        nt = [r for r in all_results if r["condition"] == label and not r["is_target"]]
        if not tr:
            continue
        avg_r = sum(r["rank"] for r in tr) / len(tr)
        med_r = sorted(r["rank"] for r in tr)[len(tr) // 2]
        avg_l = sum(r["logit"] for r in tr) / len(tr)
        if nt:
            avg_nt_l = sum(r["logit"] for r in nt) / len(nt)
            avg_nt_r = sum(r["rank"] for r in nt) / len(nt)
            ratio = avg_nt_r / max(avg_r, 1)
            print(
                f"  {label:26s} | {avg_r:8.0f} {med_r:6d} {avg_l:9.3f} | "
                f"{avg_nt_l:10.3f} {avg_nt_r:9.0f} {ratio:8.1f}x",
                flush=True,
            )
        else:
            print(
                f"  {label:26s} | {avg_r:8.0f} {med_r:6d} {avg_l:9.3f} | "
                f"{'---':>10s} {'---':>9s} {'---':>8s}",
                flush=True,
            )


if __name__ == "__main__":
    main()
