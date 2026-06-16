"""出力候補評価 — 第三段階実験。

文脈から期待される仮想新語彙が、拡張 logit 空間内で
どの順位に位置するかを評価する。

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_output_candidate.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vocabsynth.analyzer import TokenizerAnalyzer
from vocabsynth.composer import ComposeMethod, EmbeddingComposer
from vocabsynth.logit_head import VirtualLogitHead, build_virtual_logit_head
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

OUTPUT_PROMPTS: dict[str, list[dict]] = {
    "place+food": [
        {
            "template": "A famous local {food_type} from {place} is called",
            "keys": {"place": 0, "food_type": 1},
        },
        {
            "template": "The specialty food of {place} known as a {food_type} is",
            "keys": {"place": 0, "food_type": 1},
        },
        {
            "template": "In {place}, a popular {food_type} is",
            "keys": {"place": 0, "food_type": 1},
        },
    ],
    "place+structure": [
        {
            "template": "A famous {structure} in {place} is called",
            "keys": {"place": 0, "structure": 1},
        },
        {
            "template": "The iconic {structure} of {place} is",
            "keys": {"place": 0, "structure": 1},
        },
        {
            "template": "In {place}, a well-known {structure} is",
            "keys": {"place": 0, "structure": 1},
        },
    ],
}

COMPOSE_METHODS = {
    "mean": ComposeMethod.MEAN,
    "head_weighted": ComposeMethod.HEAD_WEIGHTED,
    "length_weighted": ComposeMethod.LENGTH_WEIGHTED,
}


def build_prompt(
    template: str,
    keys: dict[str, int],
    components: list[str],
) -> str:
    """プロンプトテンプレートに構成要素を埋め込む。"""
    kwargs = {}
    for key_name, comp_idx in keys.items():
        kwargs[key_name] = components[comp_idx]
    return template.format(**kwargs)


@torch.no_grad()
def evaluate_output_candidate(
    model,
    tokenizer,
    logit_head: VirtualLogitHead,
    prompt: str,
) -> list[dict]:
    """一つのプロンプトに対する全仮想語彙の順位を返す。"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    outputs = model(input_ids=input_ids, output_hidden_states=True)
    hidden_state = outputs.hidden_states[-1]
    vocab_logits = outputs.logits

    rankings = logit_head.get_virtual_rankings(hidden_state, vocab_logits)

    top10_logits = vocab_logits[0, -1, :].topk(10)
    top10 = [
        (tokenizer.decode([tid]).strip(), top10_logits.values[i].item())
        for i, tid in enumerate(top10_logits.indices.tolist())
    ]

    for r in rankings:
        r["top10_vocab"] = top10

    return rankings


def main() -> None:
    output_dir = Path("results/output_candidate")
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

    embedding_weight = model.get_input_embeddings().weight.detach()
    composer = EmbeddingComposer(embedding_weight, tokenizer)

    out_emb = model.get_output_embeddings()
    inp_emb = model.get_input_embeddings()
    is_weight_tied = out_emb.weight.data_ptr() == inp_emb.weight.data_ptr()
    print(f"Weight tying: {is_weight_tied}", flush=True)

    all_results = []
    conditions = list(COMPOSE_METHODS.items()) + [("random", None)]

    for method_name, method in conditions:
        print(f"\n{'='*70}", flush=True)
        print(f"条件: {method_name}", flush=True)
        print(f"{'='*70}", flush=True)

        if method is None:
            random_weights = []
            for vtoken in registry:
                w = composer.compose_random(vtoken.component_token_ids)
                random_weights.append(w)
            output_weights = torch.stack(random_weights).to(device)
            logit_head = VirtualLogitHead(
                output_weights, registry.surfaces()
            )
        else:
            logit_head = build_virtual_logit_head(
                model, tokenizer, registry, composer,
                method=method,
                use_lm_head_weights=not is_weight_tied,
            )

        for vtoken in registry:
            relation_key = vtoken.relation.value
            prompts_config = OUTPUT_PROMPTS.get(relation_key, [])

            for pcfg in prompts_config:
                prompt = build_prompt(
                    pcfg["template"], pcfg["keys"], vtoken.components
                )

                rankings = evaluate_output_candidate(
                    model, tokenizer, logit_head, prompt
                )

                target_ranking = next(
                    (r for r in rankings if r["surface"] == vtoken.surface),
                    None,
                )

                if target_ranking is None:
                    continue

                result = {
                    "word": vtoken.surface,
                    "relation": relation_key,
                    "method": method_name,
                    "prompt": prompt,
                    "rank": target_ranking["rank"],
                    "logit": target_ranking["logit"],
                    "probability": target_ranking["probability"],
                    "total_candidates": target_ranking["total_candidates"],
                    "top10_vocab": target_ranking.get("top10_vocab", []),
                }
                all_results.append(result)

                print(
                    f"  {vtoken.surface:18s} | "
                    f"順位={result['rank']:6d}/{result['total_candidates']} | "
                    f"logit={result['logit']:8.3f} | "
                    f"確率={result['probability']:.6f} | "
                    f"{prompt[:50]}",
                    flush=True,
                )

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

    print("\n\n=== 条件別集約 ===", flush=True)
    for method_name in [m for m, _ in conditions]:
        mr = [r for r in all_results if r["method"] == method_name]
        if not mr:
            continue
        avg_rank = sum(r["rank"] for r in mr) / len(mr)
        median_rank = sorted(r["rank"] for r in mr)[len(mr) // 2]
        avg_logit = sum(r["logit"] for r in mr) / len(mr)
        print(
            f"  {method_name:20s} | "
            f"平均順位={avg_rank:8.0f} | "
            f"中央値順位={median_rank:6d} | "
            f"平均logit={avg_logit:8.3f}",
            flush=True,
        )

    print("\n=== 対象語別集約（全条件平均）===", flush=True)
    for vtoken in registry:
        wr = [r for r in all_results if r["word"] == vtoken.surface]
        if not wr:
            continue
        avg_rank = sum(r["rank"] for r in wr) / len(wr)
        print(f"  {vtoken.surface:18s} | 平均順位={avg_rank:.0f}", flush=True)


if __name__ == "__main__":
    main()
