"""意味推論評価 — 第二段階実験。

合成トークンが構成要素間の関係を反映しているかを評価する。
たとえば place+food 型の合成トークンに対して、food/dish/specialty
などのカテゴリ候補語の確率・順位を測定する。

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_semantic_inference.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from vocabsynth.analyzer import TokenizerAnalyzer
from vocabsynth.composer import ComposeMethod, EmbeddingComposer
from vocabsynth.injector import VirtualInputInjector, build_multi_token_inputs
from vocabsynth.registry import RelationType, VocabularyRegistry

MODEL_NAME = "EleutherAI/pythia-410m"

VIRTUAL_TOKENS = [
    {"surface": "NaritaCake", "components": ["Narita", "Cake"], "relation": "place+food"},
    {"surface": "OsakaNoodle", "components": ["Osaka", "Noodle"], "relation": "place+food"},
    {"surface": "BerlinPretzel", "components": ["Berlin", "Pretzel"], "relation": "place+food"},
    {"surface": "TokyoBridge", "components": ["Tokyo", "Bridge"], "relation": "place+structure"},
    {"surface": "ParisChocolate", "components": ["Paris", "Chocolate"], "relation": "place+food"},
    {"surface": "LondonPie", "components": ["London", "Pie"], "relation": "place+food"},
]

CATEGORY_PROBES: dict[str, dict] = {
    "place+food": {
        "prompts": [
            "{word} is a famous local",
            "{word} is a type of",
            "You can eat {word} in",
            "{word} is a popular",
        ],
        "target_tokens": ["food", "dish", "snack", "dessert", "meal", "cuisine", "delicacy"],
        "anti_tokens": ["building", "bridge", "tower", "museum", "station"],
    },
    "place+structure": {
        "prompts": [
            "{word} is a famous",
            "{word} is a type of",
            "You can visit {word} in",
            "{word} is a popular",
        ],
        "target_tokens": ["bridge", "structure", "landmark", "building", "monument"],
        "anti_tokens": ["food", "dish", "snack", "dessert", "meal"],
    },
}

COMPOSE_METHODS = {
    "mean": ComposeMethod.MEAN,
    "head_weighted": ComposeMethod.HEAD_WEIGHTED,
    "length_weighted": ComposeMethod.LENGTH_WEIGHTED,
}


@torch.no_grad()
def probe_next_token(
    model,
    tokenizer,
    injector: VirtualInputInjector | None,
    text: str,
    probe_tokens: list[str],
    use_virtual: bool,
) -> dict[str, dict]:
    """次トークン位置での特定語彙の確率と順位を測定する。

    Returns:
        各探索トークンの確率、logit、全語彙中の順位を含む辞書。
    """
    device = next(model.parameters()).device

    if use_virtual and injector is not None:
        result = injector.inject(text)
        out = model(
            inputs_embeds=result.inputs_embeds,
            attention_mask=result.attention_mask,
        )
    else:
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        out = model(input_ids=input_ids)

    logits = out.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    sorted_indices = logits.argsort(descending=True)
    rank_map = torch.zeros_like(logits, dtype=torch.long)
    rank_map[sorted_indices] = torch.arange(len(sorted_indices), device=device)

    results = {}
    for token_str in probe_tokens:
        token_id = tokenizer.encode(token_str, add_special_tokens=False)
        if not token_id:
            continue
        tid = token_id[0]
        results[token_str] = {
            "token_id": tid,
            "probability": probs[tid].item(),
            "logit": logits[tid].item(),
            "rank": rank_map[tid].item(),
        }

    top5_ids = sorted_indices[:5].tolist()
    top5 = [(tokenizer.decode([tid]).strip(), probs[tid].item()) for tid in top5_ids]
    results["_top5"] = top5

    return results


def evaluate_word(
    model,
    tokenizer,
    word: str,
    relation: str,
    method_name: str,
    injector: VirtualInputInjector | None,
) -> list[dict]:
    """一つの仮想トークンの意味推論能力を評価する。"""
    probe_config = CATEGORY_PROBES.get(relation)
    if probe_config is None:
        return []

    all_probes = probe_config["target_tokens"] + probe_config["anti_tokens"]
    results = []

    for template in probe_config["prompts"]:
        prompt = template.format(word=word)

        virtual_probes = probe_next_token(
            model, tokenizer, injector, prompt, all_probes, use_virtual=True
        )
        multi_probes = probe_next_token(
            model, tokenizer, None, prompt, all_probes, use_virtual=False
        )

        target_probs_v = [
            virtual_probes[t]["probability"]
            for t in probe_config["target_tokens"]
            if t in virtual_probes
        ]
        anti_probs_v = [
            virtual_probes[t]["probability"]
            for t in probe_config["anti_tokens"]
            if t in virtual_probes
        ]
        target_probs_m = [
            multi_probes[t]["probability"]
            for t in probe_config["target_tokens"]
            if t in multi_probes
        ]
        anti_probs_m = [
            multi_probes[t]["probability"]
            for t in probe_config["anti_tokens"]
            if t in multi_probes
        ]

        target_avg_rank_v = sum(
            virtual_probes[t]["rank"]
            for t in probe_config["target_tokens"]
            if t in virtual_probes
        ) / max(len(target_probs_v), 1)
        target_avg_rank_m = sum(
            multi_probes[t]["rank"]
            for t in probe_config["target_tokens"]
            if t in multi_probes
        ) / max(len(target_probs_m), 1)

        results.append({
            "word": word,
            "relation": relation,
            "method": method_name,
            "prompt": template,
            "target_sum_prob_virtual": sum(target_probs_v),
            "target_sum_prob_multi": sum(target_probs_m),
            "anti_sum_prob_virtual": sum(anti_probs_v),
            "anti_sum_prob_multi": sum(anti_probs_m),
            "target_avg_rank_virtual": target_avg_rank_v,
            "target_avg_rank_multi": target_avg_rank_m,
            "top5_virtual": virtual_probes.get("_top5", []),
            "top5_multi": multi_probes.get("_top5", []),
            "detail_virtual": {
                k: v for k, v in virtual_probes.items() if k != "_top5"
            },
            "detail_multi": {
                k: v for k, v in multi_probes.items() if k != "_top5"
            },
        })

    return results


def main() -> None:
    output_dir = Path("results/semantic_inference")
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
    decompositions = analyzer.analyze_registry(registry)

    embedding_weight = model.get_input_embeddings().weight.detach()
    composer = EmbeddingComposer(embedding_weight, tokenizer)

    all_results = []
    conditions = list(COMPOSE_METHODS.items()) + [("random", None)]

    for method_name, method in conditions:
        print(f"\n{'='*60}", flush=True)
        print(f"条件: {method_name}", flush=True)
        print(f"{'='*60}", flush=True)

        virtual_embeddings: dict[str, torch.Tensor] = {}
        for vtoken in registry:
            if method is None:
                emb = composer.compose_random(vtoken.component_token_ids)
            else:
                emb = composer.compose(vtoken.component_token_ids, method)
            virtual_embeddings[vtoken.surface] = emb

        injector = VirtualInputInjector(
            model, tokenizer, registry, virtual_embeddings
        )

        for vtoken in registry:
            results = evaluate_word(
                model, tokenizer,
                vtoken.surface, vtoken.relation.value,
                method_name, injector,
            )
            all_results.extend(results)

            for r in results:
                print(
                    f"  {vtoken.surface:18s} | {r['prompt'][:35]:35s} | "
                    f"目標確率合計 V={r['target_sum_prob_virtual']:.4f} "
                    f"M={r['target_sum_prob_multi']:.4f} | "
                    f"目標平均順位 V={r['target_avg_rank_virtual']:.0f} "
                    f"M={r['target_avg_rank_multi']:.0f}",
                    flush=True,
                )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"

    summary = {
        "model": MODEL_NAME,
        "timestamp": timestamp,
        "virtual_tokens": VIRTUAL_TOKENS,
        "category_probes": {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in CATEGORY_PROBES.items()
        },
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)

    print("\n\n=== 条件別集約 ===", flush=True)
    for method_name in [m for m, _ in conditions]:
        mr = [r for r in all_results if r["method"] == method_name]
        if not mr:
            continue
        avg_target_v = sum(r["target_sum_prob_virtual"] for r in mr) / len(mr)
        avg_target_m = sum(r["target_sum_prob_multi"] for r in mr) / len(mr)
        avg_anti_v = sum(r["anti_sum_prob_virtual"] for r in mr) / len(mr)
        avg_rank_v = sum(r["target_avg_rank_virtual"] for r in mr) / len(mr)
        avg_rank_m = sum(r["target_avg_rank_multi"] for r in mr) / len(mr)
        print(
            f"  {method_name:20s} | "
            f"目標確率 V={avg_target_v:.4f} M={avg_target_m:.4f} | "
            f"反目標確率 V={avg_anti_v:.6f} | "
            f"目標順位 V={avg_rank_v:.0f} M={avg_rank_m:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
