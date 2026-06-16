"""入力同等性評価 — 第一段階実験。

複数トークンとして入力した場合と、合成単一トークンとして入力した場合の
次トークン分布・隠れ状態を比較する。

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_input_equiv.py
"""

from __future__ import annotations

import json
import os
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
]

PROMPT_TEMPLATES = [
    "{word} is a local specialty from",
    "{word} is famous for",
    "I tried {word} in",
]

COMPOSE_METHODS = {
    "mean": ComposeMethod.MEAN,
    "head_weighted": ComposeMethod.HEAD_WEIGHTED,
    "length_weighted": ComposeMethod.LENGTH_WEIGHTED,
}


def compute_kl_divergence(
    logits_ref: torch.Tensor, logits_test: torch.Tensor
) -> float:
    """参照分布と検証分布間の KL 発散を計算する。

    KL(P_ref || P_test) を返す。
    """
    log_p = F.log_softmax(logits_ref, dim=-1)
    log_q = F.log_softmax(logits_test, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum().item()


def compute_topk_overlap(
    logits_ref: torch.Tensor, logits_test: torch.Tensor, k: int
) -> float:
    """上位 k トークンの重複率を計算する。"""
    topk_ref = set(logits_ref.topk(k).indices.tolist())
    topk_test = set(logits_test.topk(k).indices.tolist())
    return len(topk_ref & topk_test) / k


def compute_cosine_similarity(h1: torch.Tensor, h2: torch.Tensor) -> float:
    """二つの隠れ状態間の余弦類似度を計算する。"""
    return F.cosine_similarity(h1.flatten().unsqueeze(0), h2.flatten().unsqueeze(0)).item()


@torch.no_grad()
def evaluate_single(
    model,
    tokenizer,
    injector: VirtualInputInjector,
    word: str,
    prompt_template: str,
    method_name: str,
) -> dict:
    """一つの仮想トークン・プロンプト・合成方式の組み合わせを評価する。"""
    prompt_virtual = prompt_template.format(word=word)
    prompt_multi = prompt_template.format(word=word)

    multi_embeds, multi_mask = build_multi_token_inputs(
        model, tokenizer, prompt_multi
    )
    multi_out = model(inputs_embeds=multi_embeds, attention_mask=multi_mask, output_hidden_states=True)
    multi_logits = multi_out.logits[0, -1, :]
    multi_hidden = multi_out.hidden_states[-1][0, -1, :]

    virtual_result = injector.inject(prompt_virtual)
    virtual_out = model(
        inputs_embeds=virtual_result.inputs_embeds,
        attention_mask=virtual_result.attention_mask,
        output_hidden_states=True,
    )
    virtual_logits = virtual_out.logits[0, -1, :]
    virtual_hidden = virtual_out.hidden_states[-1][0, -1, :]

    return {
        "word": word,
        "prompt": prompt_template,
        "method": method_name,
        "kl_divergence": compute_kl_divergence(multi_logits, virtual_logits),
        "topk_overlap_10": compute_topk_overlap(multi_logits, virtual_logits, 10),
        "topk_overlap_50": compute_topk_overlap(multi_logits, virtual_logits, 50),
        "cosine_similarity": compute_cosine_similarity(multi_hidden, virtual_hidden),
        "multi_token_count": multi_embeds.shape[1],
        "virtual_token_count": virtual_result.inputs_embeds.shape[1],
    }


def main() -> None:
    output_dir = Path("results/input_equiv")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"モデル読み込み: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"デバイス: {device}", flush=True)

    registry = VocabularyRegistry()
    registry.add_from_dicts(VIRTUAL_TOKENS)

    analyzer = TokenizerAnalyzer(tokenizer)
    decompositions = analyzer.analyze_registry(registry)

    print("\n=== トークン分解結果 ===", flush=True)
    for surface, decomp in decompositions.items():
        print(f"  {surface} -> {decomp.token_strings} (IDs: {decomp.token_ids})", flush=True)

    embedding_weight = model.get_input_embeddings().weight.detach()
    composer = EmbeddingComposer(embedding_weight, tokenizer)

    all_results = []

    conditions = list(COMPOSE_METHODS.items()) + [("random", None)]

    for method_name, method in conditions:
        print(f"\n=== 条件: {method_name} ===", flush=True)

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
            for template in PROMPT_TEMPLATES:
                result = evaluate_single(
                    model, tokenizer, injector,
                    vtoken.surface, template, method_name,
                )
                all_results.append(result)
                print(
                    f"  {vtoken.surface} | {template[:30]:30s} | "
                    f"KL={result['kl_divergence']:.4f} "
                    f"top10={result['topk_overlap_10']:.2f} "
                    f"top50={result['topk_overlap_50']:.2f} "
                    f"cos={result['cosine_similarity']:.4f}",
                    flush=True,
                )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"

    summary = {
        "model": MODEL_NAME,
        "timestamp": timestamp,
        "virtual_tokens": VIRTUAL_TOKENS,
        "prompt_templates": PROMPT_TEMPLATES,
        "decompositions": {
            s: {"token_ids": d.token_ids, "token_strings": d.token_strings}
            for s, d in decompositions.items()
        },
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)

    print("\n=== 条件別平均 ===", flush=True)
    for method_name in [m for m, _ in conditions]:
        method_results = [r for r in all_results if r["method"] == method_name]
        avg_kl = sum(r["kl_divergence"] for r in method_results) / len(method_results)
        avg_top10 = sum(r["topk_overlap_10"] for r in method_results) / len(method_results)
        avg_top50 = sum(r["topk_overlap_50"] for r in method_results) / len(method_results)
        avg_cos = sum(r["cosine_similarity"] for r in method_results) / len(method_results)
        print(
            f"  {method_name:20s} | KL={avg_kl:.4f} top10={avg_top10:.2f} "
            f"top50={avg_top50:.2f} cos={avg_cos:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
