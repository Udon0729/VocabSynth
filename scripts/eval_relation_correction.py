"""関係補正ベクトルの評価実験。

基本合成方式 × λ 値のグリッドで、関係補正の有無による
入力同等性・出力候補順位の変化を評価する。

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_relation_correction.py
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
from vocabsynth.corrector import RelationCorrector
from vocabsynth.injector import VirtualInputInjector, build_multi_token_inputs
from vocabsynth.logit_head import VirtualLogitHead, build_virtual_logit_head
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

INPUT_PROMPTS = [
    "{word} is a local specialty from",
    "{word} is famous for",
    "I tried {word} in",
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
    "head_weighted": ComposeMethod.HEAD_WEIGHTED,
    "length_weighted": ComposeMethod.LENGTH_WEIGHTED,
}

LAMBDAS = [0.0, 0.25, 0.5, 1.0]


def compute_kl(logits_ref: torch.Tensor, logits_test: torch.Tensor) -> float:
    log_p = F.log_softmax(logits_ref, dim=-1)
    log_q = F.log_softmax(logits_test, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum().item()


def topk_overlap(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int) -> float:
    a = set(logits_a.topk(k).indices.tolist())
    b = set(logits_b.topk(k).indices.tolist())
    return len(a & b) / k


@torch.no_grad()
def eval_input_equivalence(
    model, tokenizer, injector, word, template,
) -> dict:
    """複数トークン入力との比較指標を返す。"""
    prompt = template.format(word=word)
    multi_embeds, multi_mask = build_multi_token_inputs(model, tokenizer, prompt)
    multi_out = model(
        inputs_embeds=multi_embeds, attention_mask=multi_mask,
        output_hidden_states=True,
    )
    multi_logits = multi_out.logits[0, -1, :]
    multi_hidden = multi_out.hidden_states[-1][0, -1, :]

    virt = injector.inject(prompt)
    virt_out = model(
        inputs_embeds=virt.inputs_embeds, attention_mask=virt.attention_mask,
        output_hidden_states=True,
    )
    virt_logits = virt_out.logits[0, -1, :]
    virt_hidden = virt_out.hidden_states[-1][0, -1, :]

    return {
        "kl_divergence": compute_kl(multi_logits, virt_logits),
        "topk_overlap_10": topk_overlap(multi_logits, virt_logits, 10),
        "topk_overlap_50": topk_overlap(multi_logits, virt_logits, 50),
        "cosine_similarity": F.cosine_similarity(
            multi_hidden.unsqueeze(0), virt_hidden.unsqueeze(0)
        ).item(),
    }


@torch.no_grad()
def eval_output_candidate(
    model, tokenizer, logit_head, word, template, keys, components,
) -> dict:
    """仮想語彙の出力順位を返す。"""
    kwargs = {k: components[v] for k, v in keys.items()}
    prompt = template.format(**kwargs)

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    outputs = model(input_ids=input_ids, output_hidden_states=True)

    rankings = logit_head.get_virtual_rankings(
        outputs.hidden_states[-1], outputs.logits,
    )
    target = next((r for r in rankings if r["surface"] == word), None)
    if target is None:
        return {"rank": -1, "logit": 0.0, "probability": 0.0}
    return {
        "rank": target["rank"],
        "logit": target["logit"],
        "probability": target["probability"],
    }


def main() -> None:
    output_dir = Path("results/relation_correction")
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
    corrector = RelationCorrector(model, tokenizer, embedding_weight)

    out_emb = model.get_output_embeddings()
    inp_emb = model.get_input_embeddings()
    is_weight_tied = out_emb.weight.data_ptr() == inp_emb.weight.data_ptr()
    print(f"Weight tying: {is_weight_tied}", flush=True)

    print("\n=== 補正ベクトル診断 ===", flush=True)
    for rel in [RelationType.PLACE_FOOD, RelationType.PLACE_STRUCTURE]:
        for bm_name, bm in BASE_METHODS.items():
            diag = corrector.diagnose(rel, bm)
            print(
                f"  {rel.value:20s} | base={bm_name:15s} | "
                f"R_r_norm={diag['R_r_norm']:.2f} | "
                f"delta_mean={diag['mean_delta_norm']:.2f} ± {diag['std_delta_norm']:.2f} | "
                f"cos_pairs={diag['mean_pairwise_cosine']:.3f}",
                flush=True,
            )

    all_results = []

    for bm_name, bm in BASE_METHODS.items():
        for lam in LAMBDAS:
            cond_name = f"{bm_name}_λ{lam:.2f}"
            print(f"\n{'='*70}", flush=True)
            print(f"条件: {cond_name}", flush=True)
            print(f"{'='*70}", flush=True)

            virtual_embeddings: dict[str, torch.Tensor] = {}
            for vtoken in registry:
                base_emb = composer.compose(vtoken.component_token_ids, bm)
                corrected = corrector.apply_correction(
                    base_emb, vtoken.relation, lam, bm,
                )
                virtual_embeddings[vtoken.surface] = corrected

            injector = VirtualInputInjector(
                model, tokenizer, registry, virtual_embeddings,
            )

            output_weights = []
            for vtoken in registry:
                if is_weight_tied:
                    w = virtual_embeddings[vtoken.surface]
                else:
                    lm_weight = model.get_output_embeddings().weight.detach()
                    lm_composer = EmbeddingComposer(lm_weight, tokenizer)
                    base_w = lm_composer.compose(vtoken.component_token_ids, bm)
                    w = corrector.apply_correction(
                        base_w, vtoken.relation, lam, bm,
                    )
                output_weights.append(w)

            logit_head = VirtualLogitHead(
                torch.stack(output_weights).to(device),
                registry.surfaces(),
            )

            for vtoken in registry:
                for tmpl in INPUT_PROMPTS:
                    ie = eval_input_equivalence(
                        model, tokenizer, injector, vtoken.surface, tmpl,
                    )
                    result = {
                        "word": vtoken.surface,
                        "relation": vtoken.relation.value,
                        "method": bm_name,
                        "lambda": lam,
                        "eval_type": "input_equiv",
                        "prompt": tmpl,
                        **ie,
                    }
                    all_results.append(result)

                rel_key = vtoken.relation.value
                for tmpl, keys in OUTPUT_PROMPTS.get(rel_key, []):
                    oc = eval_output_candidate(
                        model, tokenizer, logit_head,
                        vtoken.surface, tmpl, keys, vtoken.components,
                    )
                    result = {
                        "word": vtoken.surface,
                        "relation": rel_key,
                        "method": bm_name,
                        "lambda": lam,
                        "eval_type": "output_cand",
                        "prompt": tmpl,
                        **oc,
                    }
                    all_results.append(result)

            ie_results = [r for r in all_results
                          if r["method"] == bm_name and r["lambda"] == lam
                          and r["eval_type"] == "input_equiv"]
            oc_results = [r for r in all_results
                          if r["method"] == bm_name and r["lambda"] == lam
                          and r["eval_type"] == "output_cand"]

            if ie_results:
                avg_kl = sum(r["kl_divergence"] for r in ie_results) / len(ie_results)
                avg_cos = sum(r["cosine_similarity"] for r in ie_results) / len(ie_results)
                avg_t50 = sum(r["topk_overlap_50"] for r in ie_results) / len(ie_results)
                print(
                    f"  入力同等性: KL={avg_kl:.4f} cos={avg_cos:.4f} top50={avg_t50:.2f}",
                    flush=True,
                )

            if oc_results:
                valid = [r for r in oc_results if r["rank"] >= 0]
                if valid:
                    avg_rank = sum(r["rank"] for r in valid) / len(valid)
                    med_rank = sorted(r["rank"] for r in valid)[len(valid) // 2]
                    avg_logit = sum(r["logit"] for r in valid) / len(valid)
                    print(
                        f"  出力候補:   平均順位={avg_rank:.0f} "
                        f"中央値={med_rank} 平均logit={avg_logit:.3f}",
                        flush=True,
                    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "timestamp": timestamp,
            "weight_tied": is_weight_tied,
            "lambdas": LAMBDAS,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)

    print("\n\n{'='*70}", flush=True)
    print("=== 条件別集約: 入力同等性 ===", flush=True)
    print(f"{'条件':30s} | {'KL':>8s} {'cos':>8s} {'top50':>8s}", flush=True)
    print("-" * 62, flush=True)
    for bm_name in BASE_METHODS:
        for lam in LAMBDAS:
            ie = [r for r in all_results
                  if r["method"] == bm_name and r["lambda"] == lam
                  and r["eval_type"] == "input_equiv"]
            if not ie:
                continue
            avg_kl = sum(r["kl_divergence"] for r in ie) / len(ie)
            avg_cos = sum(r["cosine_similarity"] for r in ie) / len(ie)
            avg_t50 = sum(r["topk_overlap_50"] for r in ie) / len(ie)
            tag = f"{bm_name}_λ{lam:.2f}"
            print(f"  {tag:28s} | {avg_kl:8.4f} {avg_cos:8.4f} {avg_t50:8.2f}", flush=True)

    print("\n=== 条件別集約: 出力候補順位 ===", flush=True)
    print(f"{'条件':30s} | {'平均順位':>10s} {'中央値':>8s} {'平均logit':>10s}", flush=True)
    print("-" * 66, flush=True)
    for bm_name in BASE_METHODS:
        for lam in LAMBDAS:
            oc = [r for r in all_results
                  if r["method"] == bm_name and r["lambda"] == lam
                  and r["eval_type"] == "output_cand"
                  and r.get("rank", -1) >= 0]
            if not oc:
                continue
            avg_rank = sum(r["rank"] for r in oc) / len(oc)
            med_rank = sorted(r["rank"] for r in oc)[len(oc) // 2]
            avg_logit = sum(r["logit"] for r in oc) / len(oc)
            tag = f"{bm_name}_λ{lam:.2f}"
            print(f"  {tag:28s} | {avg_rank:10.0f} {med_rank:8d} {avg_logit:10.3f}", flush=True)


if __name__ == "__main__":
    main()
