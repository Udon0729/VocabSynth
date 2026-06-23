"""内部表現抽出型出力重み合成の評価。

既存の全手法（静的合成、GW、構成語ロジット、関係修正）と
新手法（内部表現抽出型）を同一条件で比較する包括的実験。

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_contextual_head.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vocabsynth.analyzer import TokenizerAnalyzer
from vocabsynth.composer import ComposeMethod, EmbeddingComposer
from vocabsynth.contextual_head import (
    ContextualMethod,
    ContextualOutputHead,
    build_contextual_output_head,
)
from vocabsynth.corrector import DEFAULT_PAIRS, RelationCorrector
from vocabsynth.logit_head import (
    VirtualLogitHead,
    build_virtual_logit_head,
)
from vocabsynth.ot_composer import OTMethod, OTOutputComposer
from vocabsynth.registry import RelationType, VocabularyRegistry

MODEL_NAME = "EleutherAI/pythia-410m"

VIRTUAL_TOKENS = [
    {"surface": "NaritaCake", "components": ["Narita", "Cake"],
     "relation": "place+food"},
    {"surface": "OsakaNoodle", "components": ["Osaka", "Noodle"],
     "relation": "place+food"},
    {"surface": "BerlinPretzel", "components": ["Berlin", "Pretzel"],
     "relation": "place+food"},
    {"surface": "TokyoBridge", "components": ["Tokyo", "Bridge"],
     "relation": "place+structure"},
    {"surface": "ParisChocolate", "components": ["Paris", "Chocolate"],
     "relation": "place+food"},
    {"surface": "LondonPie", "components": ["London", "Pie"],
     "relation": "place+food"},
]

OUTPUT_PROMPTS: dict[str, list[dict]] = {
    "place+food": [
        {"template": "A famous local {food_type} from {place} is called",
         "keys": {"place": 0, "food_type": 1}},
        {"template": "The specialty food of {place} known as a {food_type} is",
         "keys": {"place": 0, "food_type": 1}},
        {"template": "In {place}, a popular {food_type} is",
         "keys": {"place": 0, "food_type": 1}},
    ],
    "place+structure": [
        {"template": "A famous {structure} in {place} is called",
         "keys": {"place": 0, "structure": 1}},
        {"template": "The iconic {structure} of {place} is",
         "keys": {"place": 0, "structure": 1}},
        {"template": "In {place}, a well-known {structure} is",
         "keys": {"place": 0, "structure": 1}},
    ],
}


def build_prompt(template: str, keys: dict[str, int],
                 components: list[str]) -> str:
    kwargs = {}
    for key_name, comp_idx in keys.items():
        kwargs[key_name] = components[comp_idx]
    return template.format(**kwargs)


@torch.no_grad()
def evaluate_head(model, tokenizer, head, prompt: str) -> list[dict]:
    """共通の評価関数。head は get_virtual_rankings を持つ任意のヘッド。"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    outputs = model(input_ids=input_ids, output_hidden_states=True)
    hidden_state = outputs.hidden_states[-1]
    vocab_logits = outputs.logits

    return head.get_virtual_rankings(hidden_state, vocab_logits)


def run_condition(
    model, tokenizer, head, registry, condition_name: str,
) -> list[dict]:
    """一つの条件で全仮想トークン・全プロンプトを評価する。"""
    results = []
    for vtoken in registry:
        relation_key = vtoken.relation.value
        prompts_config = OUTPUT_PROMPTS.get(relation_key, [])

        for pcfg in prompts_config:
            prompt = build_prompt(
                pcfg["template"], pcfg["keys"], vtoken.components,
            )

            rankings = evaluate_head(model, tokenizer, head, prompt)
            target = next(
                (r for r in rankings if r["surface"] == vtoken.surface),
                None,
            )
            if target is None:
                continue

            # 非対象語の順位も記録
            non_targets = [
                r for r in rankings if r["surface"] != vtoken.surface
            ]

            results.append({
                "word": vtoken.surface,
                "relation": relation_key,
                "condition": condition_name,
                "prompt": prompt,
                "rank": target["rank"],
                "logit": target["logit"],
                "probability": target["probability"],
                "total_candidates": target["total_candidates"],
                "non_target_ranks": {
                    r["surface"]: r["rank"] for r in non_targets
                },
                "non_target_logits": {
                    r["surface"]: r["logit"] for r in non_targets
                },
            })

    return results


def main() -> None:
    output_dir = Path("results/contextual_head")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"モデル読み込み: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"デバイス: {device}", flush=True)

    # レジストリ構築
    registry = VocabularyRegistry()
    registry.add_from_dicts(VIRTUAL_TOKENS)

    analyzer = TokenizerAnalyzer(tokenizer)
    analyzer.analyze_registry(registry)

    embedding_weight = model.get_input_embeddings().weight.detach()
    composer = EmbeddingComposer(embedding_weight, tokenizer)

    out_emb = model.get_output_embeddings()
    inp_emb = model.get_input_embeddings()
    is_weight_tied = out_emb.weight.data_ptr() == inp_emb.weight.data_ptr()
    print(f"重み共有: {is_weight_tied}", flush=True)

    all_results = []

    # ========================================
    # 条件1: 静的合成ベースライン (direct/mean)
    # ========================================
    print("\n" + "=" * 70, flush=True)
    print("条件: static/direct (mean)", flush=True)
    head_static = build_virtual_logit_head(
        model, tokenizer, registry, composer,
        method=ComposeMethod.MEAN,
        use_lm_head_weights=not is_weight_tied,
    )
    results = run_condition(
        model, tokenizer, head_static, registry, "static/direct",
    )
    all_results.extend(results)
    _print_summary("static/direct", results)

    # ========================================
    # 条件2: 関係修正 (lambda=0.25, mean)
    # ========================================
    print("\n" + "=" * 70, flush=True)
    print("条件: relation_corrected (lambda=0.25)", flush=True)
    corrector = RelationCorrector(model, tokenizer, embedding_weight)
    corrected_weights = []
    for vtoken in registry:
        base = composer.compose(
            vtoken.component_token_ids, ComposeMethod.MEAN,
        )
        corrected = corrector.apply_correction(
            base, vtoken.relation, lambda_=0.25,
        )
        corrected_weights.append(corrected)
    corrected_W = torch.stack(corrected_weights).to(device)
    head_corrected = VirtualLogitHead(corrected_W, registry.surfaces())
    results = run_condition(
        model, tokenizer, head_corrected, registry, "relation_corrected/0.25",
    )
    all_results.extend(results)
    _print_summary("relation_corrected/0.25", results)

    # ========================================
    # 条件3: GW 出力合成
    # ========================================
    print("\n" + "=" * 70, flush=True)
    print("条件: ot/gw", flush=True)
    output_weight = model.get_output_embeddings().weight.detach()
    ot_composer = OTOutputComposer(
        embedding_weight, output_weight,
        k_neighbors=200, epsilon=0.05,
    )
    gw_weights = []
    for vtoken in registry:
        w = ot_composer.compose(
            vtoken.component_token_ids,
            compose_method=ComposeMethod.MEAN,
            ot_method=OTMethod.GW,
        )
        gw_weights.append(w)
    gw_W = torch.stack(gw_weights).to(device)
    head_gw = VirtualLogitHead(gw_W, registry.surfaces())
    results = run_condition(
        model, tokenizer, head_gw, registry, "ot/gw",
    )
    all_results.extend(results)
    _print_summary("ot/gw", results)

    # ========================================
    # 新手法: 内部表現抽出型
    # ========================================
    for cm in ContextualMethod:
        print("\n" + "=" * 70, flush=True)
        print(f"条件: contextual/{cm.value}", flush=True)

        head_ctx = build_contextual_output_head(
            model, tokenizer, registry, method=cm,
        )

        # 診断情報の出力
        if head_ctx.layer_info:
            for info in head_ctx.layer_info:
                if "best_layer" in info:
                    print(
                        f"  {info.get('phrase', '?')}: "
                        f"最適層={info['best_layer']}",
                        flush=True,
                    )

        results = run_condition(
            model, tokenizer, head_ctx, registry,
            f"contextual/{cm.value}",
        )
        all_results.extend(results)
        _print_summary(f"contextual/{cm.value}", results)

    # ========================================
    # 結果保存
    # ========================================
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"

    summary = {
        "model": MODEL_NAME,
        "timestamp": timestamp,
        "weight_tied": is_weight_tied,
        "virtual_tokens": VIRTUAL_TOKENS,
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)

    # ========================================
    # 条件横断集約
    # ========================================
    print("\n\n" + "=" * 70, flush=True)
    print("条件横断集約", flush=True)
    print("=" * 70, flush=True)

    conditions = sorted(set(r["condition"] for r in all_results))
    print(
        f"{'条件':40s} | {'平均順位':>8s} | {'中央値':>6s} | "
        f"{'最良':>6s} | {'最悪':>6s} | {'平均logit':>10s}",
        flush=True,
    )
    print("-" * 90, flush=True)

    for cond in conditions:
        cr = [r for r in all_results if r["condition"] == cond]
        ranks = [r["rank"] for r in cr]
        logits = [r["logit"] for r in cr]
        avg_rank = sum(ranks) / len(ranks)
        median_rank = sorted(ranks)[len(ranks) // 2]
        best_rank = min(ranks)
        worst_rank = max(ranks)
        avg_logit = sum(logits) / len(logits)
        print(
            f"{cond:40s} | {avg_rank:8.0f} | {median_rank:6d} | "
            f"{best_rank:6d} | {worst_rank:6d} | {avg_logit:10.3f}",
            flush=True,
        )

    # 対象語別の条件別比較
    print("\n対象語別比較:", flush=True)
    for vtoken in registry:
        print(f"\n  {vtoken.surface}:", flush=True)
        for cond in conditions:
            cr = [
                r for r in all_results
                if r["condition"] == cond and r["word"] == vtoken.surface
            ]
            if not cr:
                continue
            ranks = [r["rank"] for r in cr]
            avg_rank = sum(ranks) / len(ranks)
            print(
                f"    {cond:38s} | 平均順位={avg_rank:8.0f} "
                f"(範囲: {min(ranks)}-{max(ranks)})",
                flush=True,
            )


def _print_summary(condition: str, results: list[dict]) -> None:
    """条件の結果をコンパクトに表示する。"""
    if not results:
        return
    for r in results:
        print(
            f"  {r['word']:18s} | "
            f"順位={r['rank']:6d} | "
            f"logit={r['logit']:8.3f} | "
            f"確率={r['probability']:.6f} | "
            f"{r['prompt'][:45]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
