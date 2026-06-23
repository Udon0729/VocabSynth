"""テンプレート寄与の切り分け実験。

multi_context 方式における関係型固有テンプレートの寄与を定量化する。

3条件を比較:
  1. relation_specific — 関係型固有テンプレート（既存の _CONTEXT_TEMPLATES）
  2. generic — 関係型非依存の汎用テンプレート（全関係型で共通の5文）
  3. minimal — テンプレートなし（"{phrase}" のみ）

指標: 仮想語彙内正解率（6語中で対象語が1位になる割合）

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_template_isolation.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vocabsynth.analyzer import TokenizerAnalyzer
from vocabsynth.contextual_head import (
    ContextualMethod,
    build_contextual_output_head,
)
from vocabsynth.registry import RelationType, VocabularyRegistry

MODEL_NAME = "EleutherAI/pythia-410m"

# --- 仮想語彙（既存の6語） ---
VIRTUAL_TOKENS = [
    {"surface": "NaritaCake", "components": ["Narita", "Cake"],
     "relation": "place+food"},
    {"surface": "OsakaNoodle", "components": ["Osaka", "Noodle"],
     "relation": "place+food"},
    {"surface": "BerlinPretzel", "components": ["Berlin", "Pretzel"],
     "relation": "place+food"},
    {"surface": "ParisChocolate", "components": ["Paris", "Chocolate"],
     "relation": "place+food"},
    {"surface": "LondonPie", "components": ["London", "Pie"],
     "relation": "place+food"},
    {"surface": "TokyoBridge", "components": ["Tokyo", "Bridge"],
     "relation": "place+structure"},
]

# --- 評価プロンプト（既存と同一） ---
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

# --- 条件2: 汎用テンプレート ---
_GENERIC_TEMPLATES: list[str] = [
    "{phrase} is well known",
    "The thing called {phrase} is",
    "{phrase} is a notable example",
    "People know about {phrase} because",
    "{phrase} is something special",
]

# --- 条件3: 最小テンプレート ---
_MINIMAL_TEMPLATES: list[str] = [
    "{phrase}",
]


def _build_uniform_template_dict(
    registry: VocabularyRegistry,
    templates: list[str],
) -> dict[str, list[str]]:
    """レジストリに含まれる全関係型に対して同一テンプレートを割り当てる。"""
    relation_keys: set[str] = set()
    for vtoken in registry:
        relation_keys.add(vtoken.relation.value)
    return {key: list(templates) for key in relation_keys}


def build_prompt(
    template: str, keys: dict[str, int], components: list[str],
) -> str:
    kwargs = {}
    for key_name, comp_idx in keys.items():
        kwargs[key_name] = components[comp_idx]
    return template.format(**kwargs)


@torch.no_grad()
def evaluate_virtual_accuracy(
    model, tokenizer, head, registry,
) -> list[dict]:
    """仮想語彙内正解率を評価する。

    各仮想トークン・各プロンプトについて、仮想語彙6語のうち
    対象語のロジットが1位かどうかを判定する。
    """
    device = next(model.parameters()).device
    results = []

    for vtoken in registry:
        relation_key = vtoken.relation.value
        prompts_config = OUTPUT_PROMPTS.get(relation_key, [])

        for pcfg in prompts_config:
            prompt = build_prompt(
                pcfg["template"], pcfg["keys"], vtoken.components,
            )

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            outputs = model(
                input_ids=input_ids, output_hidden_states=True,
            )
            hidden_state = outputs.hidden_states[-1]
            vocab_logits = outputs.logits

            rankings = head.get_virtual_rankings(
                hidden_state, vocab_logits,
            )

            # 仮想語彙内での順位を計算
            logit_map = {r["surface"]: r["logit"] for r in rankings}
            sorted_by_logit = sorted(
                logit_map.items(), key=lambda x: x[1], reverse=True,
            )
            virtual_rank = next(
                i for i, (name, _) in enumerate(sorted_by_logit)
                if name == vtoken.surface
            )
            is_top1 = virtual_rank == 0

            target_info = next(
                r for r in rankings if r["surface"] == vtoken.surface
            )

            results.append({
                "word": vtoken.surface,
                "relation": relation_key,
                "prompt": prompt,
                "virtual_rank": virtual_rank,
                "is_top1": is_top1,
                "logit": target_info["logit"],
                "probability": target_info["probability"],
                "global_rank": target_info["rank"],
                "virtual_logits": {
                    name: logit for name, logit in sorted_by_logit
                },
            })

    return results


def _summarize(condition_name: str, results: list[dict]) -> dict:
    """条件の集約統計量を計算する。"""
    n = len(results)
    top1_count = sum(1 for r in results if r["is_top1"])
    accuracy = top1_count / n if n > 0 else 0.0
    avg_virtual_rank = (
        sum(r["virtual_rank"] for r in results) / n if n > 0 else float("nan")
    )
    avg_global_rank = (
        sum(r["global_rank"] for r in results) / n if n > 0 else float("nan")
    )
    return {
        "condition": condition_name,
        "num_prompts": n,
        "top1_count": top1_count,
        "accuracy": accuracy,
        "avg_virtual_rank": avg_virtual_rank,
        "avg_global_rank": avg_global_rank,
    }


def main() -> None:
    output_dir = Path("results/template_isolation")
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

    all_results: dict[str, list[dict]] = {}
    summaries: list[dict] = []

    # ==================================================
    # 条件1: relation_specific（関係型固有テンプレート）
    # ==================================================
    cond_name = "relation_specific"
    print(f"\n{'=' * 70}", flush=True)
    print(f"条件: {cond_name}（関係型固有テンプレート）", flush=True)
    print("=" * 70, flush=True)

    head_specific = build_contextual_output_head(
        model, tokenizer, registry,
        method=ContextualMethod.MULTI_CONTEXT,
        context_templates=None,  # 既定の _CONTEXT_TEMPLATES を使用
    )
    results = evaluate_virtual_accuracy(
        model, tokenizer, head_specific, registry,
    )
    all_results[cond_name] = results
    summary = _summarize(cond_name, results)
    summaries.append(summary)
    _print_condition_results(cond_name, results, summary)

    # ==================================================
    # 条件2: generic（汎用テンプレート）
    # ==================================================
    cond_name = "generic"
    print(f"\n{'=' * 70}", flush=True)
    print(f"条件: {cond_name}（汎用テンプレート）", flush=True)
    print("=" * 70, flush=True)

    generic_dict = _build_uniform_template_dict(
        registry, _GENERIC_TEMPLATES,
    )
    head_generic = build_contextual_output_head(
        model, tokenizer, registry,
        method=ContextualMethod.MULTI_CONTEXT,
        context_templates=generic_dict,
    )
    results = evaluate_virtual_accuracy(
        model, tokenizer, head_generic, registry,
    )
    all_results[cond_name] = results
    summary = _summarize(cond_name, results)
    summaries.append(summary)
    _print_condition_results(cond_name, results, summary)

    # ==================================================
    # 条件3: minimal（フレーズのみ）
    # ==================================================
    cond_name = "minimal"
    print(f"\n{'=' * 70}", flush=True)
    print(f"条件: {cond_name}（フレーズのみ）", flush=True)
    print("=" * 70, flush=True)

    minimal_dict = _build_uniform_template_dict(
        registry, _MINIMAL_TEMPLATES,
    )
    head_minimal = build_contextual_output_head(
        model, tokenizer, registry,
        method=ContextualMethod.MULTI_CONTEXT,
        context_templates=minimal_dict,
    )
    results = evaluate_virtual_accuracy(
        model, tokenizer, head_minimal, registry,
    )
    all_results[cond_name] = results
    summary = _summarize(cond_name, results)
    summaries.append(summary)
    _print_condition_results(cond_name, results, summary)

    # ==================================================
    # 3条件横断比較
    # ==================================================
    print(f"\n\n{'=' * 70}", flush=True)
    print("3条件横断比較", flush=True)
    print("=" * 70, flush=True)

    print(
        f"{'条件':25s} | {'正解率':>8s} | "
        f"{'正解数':>6s} | {'総数':>4s} | "
        f"{'平均仮想順位':>12s} | {'平均全体順位':>12s}",
        flush=True,
    )
    print("-" * 85, flush=True)

    for s in summaries:
        print(
            f"{s['condition']:25s} | {s['accuracy']:8.3f} | "
            f"{s['top1_count']:6d} | {s['num_prompts']:4d} | "
            f"{s['avg_virtual_rank']:12.2f} | {s['avg_global_rank']:12.0f}",
            flush=True,
        )

    # 対象語別の条件横断比較
    print(f"\n対象語別比較:", flush=True)
    for vtoken in registry:
        print(f"\n  {vtoken.surface}:", flush=True)
        for cond_name, cond_results in all_results.items():
            word_results = [
                r for r in cond_results if r["word"] == vtoken.surface
            ]
            if not word_results:
                continue
            top1_count = sum(1 for r in word_results if r["is_top1"])
            total = len(word_results)
            acc = top1_count / total
            avg_vr = sum(r["virtual_rank"] for r in word_results) / total
            print(
                f"    {cond_name:23s} | "
                f"正解率={acc:.3f} ({top1_count}/{total}) | "
                f"平均仮想順位={avg_vr:.2f}",
                flush=True,
            )

    # ==================================================
    # テンプレート寄与の定量化
    # ==================================================
    print(f"\n\n{'=' * 70}", flush=True)
    print("テンプレート寄与の定量化", flush=True)
    print("=" * 70, flush=True)

    s_specific = next(s for s in summaries if s["condition"] == "relation_specific")
    s_generic = next(s for s in summaries if s["condition"] == "generic")
    s_minimal = next(s for s in summaries if s["condition"] == "minimal")

    delta_generic = s_specific["accuracy"] - s_generic["accuracy"]
    delta_minimal = s_specific["accuracy"] - s_minimal["accuracy"]
    delta_gen_min = s_generic["accuracy"] - s_minimal["accuracy"]

    print(f"関係型固有 - 汎用:     正解率差 = {delta_generic:+.3f}", flush=True)
    print(f"関係型固有 - 最小:     正解率差 = {delta_minimal:+.3f}", flush=True)
    print(f"汎用       - 最小:     正解率差 = {delta_gen_min:+.3f}", flush=True)
    print(flush=True)
    print(
        "解釈: 「関係型固有 - 汎用」が大きければ関係型テンプレートの"
        "固有情報が重要。", flush=True,
    )
    print(
        "      「汎用 - 最小」が大きければテンプレートによる文脈付与"
        "自体が重要。", flush=True,
    )

    # ==================================================
    # 結果保存
    # ==================================================
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"

    save_data = {
        "model": MODEL_NAME,
        "timestamp": timestamp,
        "virtual_tokens": VIRTUAL_TOKENS,
        "conditions": {
            cond: {
                "summary": next(
                    s for s in summaries if s["condition"] == cond
                ),
                "detail": cond_results,
            }
            for cond, cond_results in all_results.items()
        },
        "deltas": {
            "relation_specific_minus_generic": delta_generic,
            "relation_specific_minus_minimal": delta_minimal,
            "generic_minus_minimal": delta_gen_min,
        },
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)


def _print_condition_results(
    condition: str, results: list[dict], summary: dict,
) -> None:
    """条件ごとの詳細結果を表示する。"""
    for r in results:
        mark = "○" if r["is_top1"] else "×"
        print(
            f"  {mark} {r['word']:18s} | "
            f"仮想順位={r['virtual_rank']} | "
            f"ロジット={r['logit']:8.3f} | "
            f"全体順位={r['global_rank']:6d} | "
            f"{r['prompt'][:45]}",
            flush=True,
        )
    print(
        f"  → 正解率: {summary['accuracy']:.3f} "
        f"({summary['top1_count']}/{summary['num_prompts']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
