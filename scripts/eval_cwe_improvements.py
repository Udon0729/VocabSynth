"""CWE の弁別性改善手法の比較実験。

purpose+container と material+artifact のカテゴリ間混同を解消するため、
3つの改善手法を検証する。

1. phrase_pos: フレーズ末位置の隠れ状態を抽出（文末ではなく）
2. mean_center: 全仮想語彙の出力重みから大域平均を減算
3. differential: h(テンプレート+フレーズ) - h(テンプレート+中立語) で
   フレーズ固有の寄与を分離

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_cwe_improvements.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from vocabsynth.contextual_head import (
    ContextualOutputHead,
    _norm_match_to_vocab,
    calibrate_output_head,
)
from vocabsynth.registry import RelationType, VocabularyRegistry

MODEL_NAME = "EleutherAI/pythia-410m"

VIRTUAL_TOKENS = [
    {"surface": "NaritaCake", "components": ["Narita", "Cake"], "relation": "place+food"},
    {"surface": "OsakaNoodle", "components": ["Osaka", "Noodle"], "relation": "place+food"},
    {"surface": "BerlinPretzel", "components": ["Berlin", "Pretzel"], "relation": "place+food"},
    {"surface": "ParisChocolate", "components": ["Paris", "Chocolate"], "relation": "place+food"},
    {"surface": "LondonPie", "components": ["London", "Pie"], "relation": "place+food"},
    {"surface": "TokyoBridge", "components": ["Tokyo", "Bridge"], "relation": "place+structure"},
    {"surface": "SeoulTower", "components": ["Seoul", "Tower"], "relation": "place+structure"},
    {"surface": "CairoTemple", "components": ["Cairo", "Temple"], "relation": "place+structure"},
    {"surface": "JadeSculpture", "components": ["Jade", "Sculpture"], "relation": "material+artifact"},
    {"surface": "SilverBracelet", "components": ["Silver", "Bracelet"], "relation": "material+artifact"},
    {"surface": "CopperVase", "components": ["Copper", "Vase"], "relation": "material+artifact"},
    {"surface": "GoldRing", "components": ["Gold", "Ring"], "relation": "material+artifact"},
    {"surface": "BronzeBell", "components": ["Bronze", "Bell"], "relation": "material+artifact"},
    {"surface": "GlassWindow", "components": ["Glass", "Window"], "relation": "material+artifact"},
    {"surface": "SpiceJar", "components": ["Spice", "Jar"], "relation": "purpose+container"},
    {"surface": "WineBarrel", "components": ["Wine", "Barrel"], "relation": "purpose+container"},
    {"surface": "InkBottle", "components": ["Ink", "Bottle"], "relation": "purpose+container"},
    {"surface": "GrainSilo", "components": ["Grain", "Silo"], "relation": "purpose+container"},
    {"surface": "OilLamp", "components": ["Oil", "Lamp"], "relation": "purpose+container"},
    {"surface": "TeaPot", "components": ["Tea", "Pot"], "relation": "purpose+container"},
]

GENERIC_TEMPLATES = [
    "{phrase} is well known",
    "The thing called {phrase} is",
    "{phrase} is a notable example",
    "People know about {phrase} because",
    "{phrase} is something special",
]

EVAL_PROMPTS = {
    "place+food": [
        ("In {place}, a popular {food} is", {"place": 0, "food": 1}),
        ("A famous local {food} from {place} is called", {"place": 0, "food": 1}),
        ("The specialty food of {place} known as a {food} is", {"place": 0, "food": 1}),
    ],
    "place+structure": [
        ("In {place}, a well-known {struct} is", {"place": 0, "struct": 1}),
        ("A famous {struct} in {place} is called", {"place": 0, "struct": 1}),
        ("The iconic {struct} of {place} is", {"place": 0, "struct": 1}),
    ],
    "material+artifact": [
        ("A {artifact} made of {material} is called", {"material": 0, "artifact": 1}),
        ("The {material} {artifact} is known as", {"material": 0, "artifact": 1}),
        ("A famous {material} {artifact} is", {"material": 0, "artifact": 1}),
    ],
    "purpose+container": [
        ("A {container} designed for storing {purpose} is called", {"purpose": 0, "container": 1}),
        ("The {container} used for {purpose} is known as", {"purpose": 0, "container": 1}),
        ("A specialized {purpose} {container} is", {"purpose": 0, "container": 1}),
    ],
}

NEUTRAL_WORD = "thing"


def get_final_layer_norm(model):
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.final_layer_norm
    if hasattr(model, "transformer"):
        return model.transformer.ln_f
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    return None


def find_phrase_end_position(tokenizer, full_text, phrase):
    """テンプレート内でフレーズが終わるトークン位置を特定する。"""
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    phrase_ids = tokenizer.encode(phrase, add_special_tokens=False)

    for start in range(len(full_ids) - len(phrase_ids) + 1):
        if full_ids[start:start + len(phrase_ids)] == phrase_ids:
            return start + len(phrase_ids) - 1

    # 完全一致が見つからない場合、部分一致を試みる
    phrase_text = tokenizer.decode(phrase_ids)
    for start in range(len(full_ids)):
        for end in range(start + 1, min(start + len(phrase_ids) + 2, len(full_ids) + 1)):
            candidate = tokenizer.decode(full_ids[start:end])
            if candidate.strip() == phrase_text.strip():
                return end - 1
    return -1


@torch.no_grad()
def extract_at_position(model, tokenizer, text, position, norm_layer):
    """指定位置の正規化済み隠れ状態を抽出する。"""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    outputs = model(input_ids=input_ids, output_hidden_states=True)
    h = outputs.hidden_states[-1][0, position, :]
    if norm_layer is not None:
        h = norm_layer(h.unsqueeze(0)).squeeze(0)
    return h


@torch.no_grad()
def extract_at_last(model, tokenizer, text, norm_layer):
    """文末位置の正規化済み隠れ状態を抽出する。"""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    outputs = model(input_ids=input_ids, output_hidden_states=True)
    h = outputs.hidden_states[-1][0, -1, :]
    if norm_layer is not None:
        h = norm_layer(h.unsqueeze(0)).squeeze(0)
    return h


@torch.no_grad()
def build_heads(model, tokenizer, registry):
    """4条件の出力ヘッドを構築する。"""
    device = next(model.parameters()).device
    output_weight = model.get_output_embeddings().weight.detach()
    norm_layer = get_final_layer_norm(model)

    heads = {}
    surfaces = [vt.surface for vt in registry]

    # === 条件1: baseline（現行 multi_context、文末位置）===
    rows_baseline = []
    for vtoken in registry:
        phrase = " ".join(vtoken.components)
        hs = []
        for tmpl in GENERIC_TEMPLATES:
            text = tmpl.format(phrase=phrase)
            h = extract_at_last(model, tokenizer, text, norm_layer)
            hs.append(h)
        h_avg = torch.stack(hs).mean(dim=0)
        h_avg = _norm_match_to_vocab(h_avg, output_weight)
        rows_baseline.append(h_avg)
    W_baseline = torch.stack(rows_baseline).to(device)
    heads["baseline"] = ContextualOutputHead(W_baseline, surfaces)

    # === 条件2: phrase_pos（フレーズ末位置）===
    rows_phrase = []
    for vtoken in registry:
        phrase = " ".join(vtoken.components)
        hs = []
        for tmpl in GENERIC_TEMPLATES:
            text = tmpl.format(phrase=phrase)
            pos = find_phrase_end_position(tokenizer, text, phrase)
            if pos >= 0:
                h = extract_at_position(model, tokenizer, text, pos, norm_layer)
            else:
                h = extract_at_last(model, tokenizer, text, norm_layer)
            hs.append(h)
        h_avg = torch.stack(hs).mean(dim=0)
        h_avg = _norm_match_to_vocab(h_avg, output_weight)
        rows_phrase.append(h_avg)
    W_phrase = torch.stack(rows_phrase).to(device)
    heads["phrase_pos"] = ContextualOutputHead(W_phrase, surfaces)

    # === 条件3: mean_center（大域平均除去）===
    W_centered = W_baseline - W_baseline.mean(dim=0, keepdim=True)
    # ノルム補正（行ごと）
    vocab_median_norm = output_weight.norm(dim=1).median()
    norms = W_centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
    W_centered = W_centered * (vocab_median_norm / norms)
    heads["mean_center"] = ContextualOutputHead(W_centered, surfaces)

    # === 条件4: differential（差分符号化）===
    rows_diff = []
    for vtoken in registry:
        phrase = " ".join(vtoken.components)
        diffs = []
        for tmpl in GENERIC_TEMPLATES:
            text_with = tmpl.format(phrase=phrase)
            text_without = tmpl.format(phrase=NEUTRAL_WORD)
            h_with = extract_at_last(model, tokenizer, text_with, norm_layer)
            h_without = extract_at_last(model, tokenizer, text_without, norm_layer)
            diffs.append(h_with - h_without)
        h_diff = torch.stack(diffs).mean(dim=0)
        h_diff = _norm_match_to_vocab(h_diff, output_weight)
        rows_diff.append(h_diff)
    W_diff = torch.stack(rows_diff).to(device)
    heads["differential"] = ContextualOutputHead(W_diff, surfaces)

    # === 条件5: phrase_pos + mean_center ===
    W_phrase_centered = W_phrase - W_phrase.mean(dim=0, keepdim=True)
    norms = W_phrase_centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
    W_phrase_centered = W_phrase_centered * (vocab_median_norm / norms)
    heads["phrase_pos+center"] = ContextualOutputHead(W_phrase_centered, surfaces)

    # === 条件6: differential + mean_center ===
    W_diff_centered = W_diff - W_diff.mean(dim=0, keepdim=True)
    norms = W_diff_centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
    W_diff_centered = W_diff_centered * (vocab_median_norm / norms)
    heads["differential+center"] = ContextualOutputHead(W_diff_centered, surfaces)

    # キャリブレーション
    calibrated_heads = {}
    for name, head in heads.items():
        calibrated_heads[name] = calibrate_output_head(model, tokenizer, head)

    return calibrated_heads


@torch.no_grad()
def evaluate(model, tokenizer, heads, registry):
    """全条件を評価する。"""
    device = next(model.parameters()).device
    results = defaultdict(list)

    for vtoken in registry:
        rel_key = vtoken.relation.value
        prompts_cfg = EVAL_PROMPTS.get(rel_key, [])

        for tmpl, keys in prompts_cfg:
            kwargs = {k: vtoken.components[v] for k, v in keys.items()}
            prompt = tmpl.format(**kwargs)

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            hidden_state = outputs.hidden_states[-1]
            vocab_logits = outputs.logits

            for cond_name, head in heads.items():
                rankings = head.get_virtual_rankings(hidden_state, vocab_logits)
                target = next(
                    r for r in rankings if r["surface"] == vtoken.surface
                )
                top1 = max(rankings, key=lambda r: r["logit"])

                results[cond_name].append({
                    "word": vtoken.surface,
                    "relation": rel_key,
                    "prompt": prompt,
                    "target_logit": target["logit"],
                    "target_rank": sum(
                        1 for r in rankings if r["logit"] > target["logit"]
                    ),
                    "top1": top1["surface"],
                    "top1_logit": top1["logit"],
                    "correct": top1["surface"] == vtoken.surface,
                })

    return results


def print_results(results):
    """結果を表示する。"""
    print("\n=== 全条件比較 ===", flush=True)
    print(
        f"{'条件':25s} | {'全体':>8s} | {'place系':>8s} | "
        f"{'mat+art':>8s} | {'pur+con':>8s}",
        flush=True,
    )
    print("-" * 75, flush=True)

    for cond_name, entries in sorted(results.items()):
        total = len(entries)
        correct = sum(1 for e in entries if e["correct"])

        place = [e for e in entries if e["relation"].startswith("place")]
        mat = [e for e in entries if e["relation"] == "material+artifact"]
        pur = [e for e in entries if e["relation"] == "purpose+container"]

        place_c = sum(1 for e in place if e["correct"])
        mat_c = sum(1 for e in mat if e["correct"])
        pur_c = sum(1 for e in pur if e["correct"])

        print(
            f"  {cond_name:23s} | "
            f"{correct:2d}/{total:2d} ({100*correct/total:4.0f}%) | "
            f"{place_c:2d}/{len(place):2d} ({100*place_c/max(len(place),1):4.0f}%) | "
            f"{mat_c:2d}/{len(mat):2d} ({100*mat_c/max(len(mat),1):4.0f}%) | "
            f"{pur_c:2d}/{len(pur):2d} ({100*pur_c/max(len(pur),1):4.0f}%)",
            flush=True,
        )

    # purpose+container の詳細
    print("\n=== purpose+container 詳細 ===", flush=True)
    for cond_name, entries in sorted(results.items()):
        pur = [e for e in entries if e["relation"] == "purpose+container"]
        pur_c = sum(1 for e in pur if e["correct"])
        print(f"\n  --- {cond_name}: {pur_c}/{len(pur)} ---", flush=True)
        for e in sorted(pur, key=lambda x: x["word"]):
            mark = "○" if e["correct"] else "×"
            margin = e["target_logit"] - e["top1_logit"]
            winner_info = "" if e["correct"] else f" → 1位={e['top1']}"
            print(
                f"    {mark} {e['word']:14s} logit={e['target_logit']:7.2f}"
                f" 差={margin:+.2f}{winner_info}"
                f" | {e['prompt'][:45]}",
                flush=True,
            )

    # material+artifact の詳細
    print("\n=== material+artifact 詳細 ===", flush=True)
    for cond_name, entries in sorted(results.items()):
        mat = [e for e in entries if e["relation"] == "material+artifact"]
        mat_c = sum(1 for e in mat if e["correct"])
        print(f"\n  --- {cond_name}: {mat_c}/{len(mat)} ---", flush=True)
        for e in sorted(mat, key=lambda x: x["word"]):
            mark = "○" if e["correct"] else "×"
            margin = e["target_logit"] - e["top1_logit"]
            winner_info = "" if e["correct"] else f" → 1位={e['top1']}"
            print(
                f"    {mark} {e['word']:14s} logit={e['target_logit']:7.2f}"
                f" 差={margin:+.2f}{winner_info}"
                f" | {e['prompt'][:45]}",
                flush=True,
            )

    # 誤答のカテゴリ間流出分析
    print("\n=== カテゴリ間流出分析 ===", flush=True)
    word_to_rel = {}
    for entries in results.values():
        for e in entries:
            word_to_rel[e["word"]] = e["relation"]

    for cond_name, entries in sorted(results.items()):
        wrong = [e for e in entries if not e["correct"]]
        if not wrong:
            print(f"  {cond_name}: 誤答なし", flush=True)
            continue

        same_cat = sum(
            1 for e in wrong
            if word_to_rel.get(e["top1"], "") == e["relation"]
        )
        cross_cat = len(wrong) - same_cat
        print(
            f"  {cond_name}: 誤答{len(wrong)}件"
            f" (同カテゴリ:{same_cat} 他カテゴリ:{cross_cat})",
            flush=True,
        )


def main():
    output_dir = Path("results/cwe_improvements")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"モデル読み込み: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"デバイス: {device}", flush=True)

    registry = VocabularyRegistry()
    registry.add_from_dicts(VIRTUAL_TOKENS)

    print(f"仮想語彙: {len(VIRTUAL_TOKENS)}語", flush=True)
    for rel in ["place+food", "place+structure", "material+artifact", "purpose+container"]:
        count = sum(1 for v in VIRTUAL_TOKENS if v["relation"] == rel)
        print(f"  {rel}: {count}語", flush=True)

    print("\n出力ヘッド構築中...", flush=True)
    heads = build_heads(model, tokenizer, registry)
    print(f"条件数: {len(heads)}", flush=True)

    print("\n評価実行中...", flush=True)
    results = evaluate(model, tokenizer, heads, registry)

    print_results(results)

    # 保存
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"results_{timestamp}.json"
    serializable = {
        cond: entries for cond, entries in results.items()
    }
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {save_path}", flush=True)


if __name__ == "__main__":
    main()
