"""UOT 重心射影による出力重み合成の評価。

直接合成（ベースライン）、均衡 OT、不均衡 OT（τ 複数値）を比較し、
出力候補順位の改善を検証する。

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_ot_output.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vocabsynth.analyzer import TokenizerAnalyzer
from vocabsynth.composer import ComposeMethod, EmbeddingComposer
from vocabsynth.injector import VirtualInputInjector, build_multi_token_inputs
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

BASE_METHODS = {
    "mean": ComposeMethod.MEAN,
    "length_weighted": ComposeMethod.LENGTH_WEIGHTED,
}

K_NEIGHBORS = 200
EPSILON = 0.05
TAUS = [0.1, 0.5, 1.0, 5.0]


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


def main() -> None:
    output_dir = Path("results/ot_output")
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
    print(f"E 平均ノルム: {E.norm(dim=1).mean():.4f}", flush=True)
    print(f"W 平均ノルム: {W.norm(dim=1).mean():.4f}", flush=True)

    ot_composer = OTOutputComposer(
        E, W, k_neighbors=K_NEIGHBORS, epsilon=EPSILON, tau=1.0,
    )

    print("\n=== 近傍幾何診断 ===", flush=True)
    for vtoken in registry:
        for tid in vtoken.component_token_ids:
            diag = ot_composer.diagnose_local_geometry(tid)
            tok_str = tokenizer.decode([tid]).strip()
            print(
                f"  {tok_str:12s} (id={tid:5d}) | "
                f"E-W cos={diag['paired_cosine_mean']:.3f}±{diag['paired_cosine_std']:.3f} | "
                f"対角コスト={diag['diagonal_cost_mean']:.4f} "
                f"非対角コスト={diag['off_diagonal_cost_mean']:.4f}",
                flush=True,
            )

    all_experiment_results = []

    conditions = []
    for bm_name, bm in BASE_METHODS.items():
        conditions.append(("direct", bm_name, bm, None, None))
        conditions.append(("balanced", bm_name, bm, OTMethod.BALANCED, None))
        for tau in TAUS:
            conditions.append((f"uot_τ{tau}", bm_name, bm, OTMethod.UNBALANCED, tau))

    for ot_name, bm_name, bm, ot_method, tau in conditions:
        cond_label = f"{bm_name}/{ot_name}"
        print(f"\n{'='*70}", flush=True)
        print(f"条件: {cond_label}", flush=True)
        print(f"{'='*70}", flush=True)

        output_weight_rows = []
        for vtoken in registry:
            if ot_method is None:
                w = ot_composer.compose_direct(
                    vtoken.component_token_ids, bm, tokenizer,
                )
            else:
                if tau is not None and ot_method == OTMethod.UNBALANCED:
                    ot_composer._tau = tau
                    ot_composer._bary_cache.clear()
                w = ot_composer.compose(
                    vtoken.component_token_ids, bm, ot_method, tokenizer,
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
        avg_rank = sum(r["rank"] for r in target_results) / max(len(target_results), 1)
        med_rank = sorted(r["rank"] for r in target_results)[len(target_results) // 2]
        avg_logit = sum(r["logit"] for r in target_results) / max(len(target_results), 1)

        print(
            f"  対象語 平均順位={avg_rank:.0f} 中央値={med_rank} "
            f"平均logit={avg_logit:.3f}",
            flush=True,
        )

        for vtoken in registry:
            vr = [r for r in target_results if r["target_word"] == vtoken.surface]
            if vr:
                v_avg = sum(r["rank"] for r in vr) / len(vr)
                print(f"    {vtoken.surface:18s} 平均順位={v_avg:.0f}", flush=True)

        for r in results:
            r["condition"] = cond_label
            r["ot_method"] = ot_name
            r["base_method"] = bm_name
            r["tau"] = tau
        all_experiment_results.extend(results)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "timestamp": timestamp,
            "k_neighbors": K_NEIGHBORS,
            "epsilon": EPSILON,
            "taus": TAUS,
            "results": all_experiment_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)

    print("\n\n=== 全条件比較（対象語の順位） ===", flush=True)
    print(f"{'条件':35s} | {'平均順位':>10s} {'中央値':>8s} {'平均logit':>10s}", flush=True)
    print("-" * 72, flush=True)

    seen = set()
    for ot_name, bm_name, bm, ot_method, tau in conditions:
        cond_label = f"{bm_name}/{ot_name}"
        if cond_label in seen:
            continue
        seen.add(cond_label)

        tr = [r for r in all_experiment_results
              if r["condition"] == cond_label and r["is_target"]]
        if not tr:
            continue
        avg_r = sum(r["rank"] for r in tr) / len(tr)
        med_r = sorted(r["rank"] for r in tr)[len(tr) // 2]
        avg_l = sum(r["logit"] for r in tr) / len(tr)
        print(f"  {cond_label:33s} | {avg_r:10.0f} {med_r:8d} {avg_l:10.3f}", flush=True)

    print("\n=== 非対象語のlogit膨張チェック ===", flush=True)
    for ot_name, bm_name, bm, ot_method, tau in conditions:
        cond_label = f"{bm_name}/{ot_name}"
        if cond_label in seen and ot_name != "direct":
            non_target = [r for r in all_experiment_results
                          if r["condition"] == cond_label and not r["is_target"]]
            if non_target:
                avg_nt_logit = sum(r["logit"] for r in non_target) / len(non_target)
                avg_nt_rank = sum(r["rank"] for r in non_target) / len(non_target)
                print(
                    f"  {cond_label:33s} | 非対象: 平均logit={avg_nt_logit:.3f} "
                    f"平均順位={avg_nt_rank:.0f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
