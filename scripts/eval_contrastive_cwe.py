"""対照抽出による CWE 改善実験。

各仮想トークンの出力重みを以下で構築する:
  W_v = h_pos - α · h_neg
ここで h_pos は自カテゴリテンプレートの隠れ状態平均、
h_neg は他カテゴリテンプレートの隠れ状態平均。

これにより、カテゴリ間で共有される「物理的対象」等の
共通成分が除去され、カテゴリ固有の方向が強調される。

α の値を 0.0（ベースライン）、0.25、0.5、0.75、1.0 で比較。

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_contrastive_cwe.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vocabsynth.analyzer import TokenizerAnalyzer
from vocabsynth.contextual_head import (
    ContextualOutputHead,
    _norm_match_to_vocab,
    calibrate_output_head,
)
from vocabsynth.registry import VocabularyRegistry

MODEL_NAME = "EleutherAI/pythia-410m"

CONTEXT_TEMPLATES = {
    "place+food": [
        "{phrase} is a famous local food",
        "The specialty called {phrase} is",
        "People love eating {phrase} when visiting",
        "A popular dish known as {phrase} is",
        "{phrase} is a regional delicacy from",
    ],
    "place+structure": [
        "{phrase} is a famous landmark",
        "The structure known as {phrase} is",
        "Tourists visit {phrase} to see",
        "A well-known monument called {phrase} is",
        "{phrase} is an iconic structure in",
    ],
    "place+institution": [
        "{phrase} is a well-known institution",
        "The institution called {phrase} is located in",
        "Students attend {phrase} to study",
        "A prestigious organization known as {phrase} is",
        "{phrase} is a major institution in",
    ],
    "material+artifact": [
        "{phrase} is a famous object made of",
        "The artifact known as {phrase} is crafted from",
        "Collectors value {phrase} for its",
        "A renowned piece called {phrase} is",
        "{phrase} is a celebrated work of",
    ],
    "purpose+container": [
        "{phrase} is a container designed for",
        "The vessel known as {phrase} is used to store",
        "People use {phrase} to hold",
        "A specialized container called {phrase} is",
        "{phrase} is commonly used for storing",
    ],
}

VOCAB_PLACE_FOOD = [
    {"surface": "NaritaCake", "components": ["Narita", "Cake"], "relation": "place+food"},
    {"surface": "OsakaNoodle", "components": ["Osaka", "Noodle"], "relation": "place+food"},
    {"surface": "BerlinPretzel", "components": ["Berlin", "Pretzel"], "relation": "place+food"},
    {"surface": "ParisChocolate", "components": ["Paris", "Chocolate"], "relation": "place+food"},
    {"surface": "LondonPie", "components": ["London", "Pie"], "relation": "place+food"},
    {"surface": "RomePasta", "components": ["Rome", "Pasta"], "relation": "place+food"},
    {"surface": "SydneyCurry", "components": ["Sydney", "Curry"], "relation": "place+food"},
    {"surface": "CairoDumpling", "components": ["Cairo", "Dumpling"], "relation": "place+food"},
    {"surface": "SeoulBread", "components": ["Seoul", "Bread"], "relation": "place+food"},
    {"surface": "ViennaCheese", "components": ["Vienna", "Cheese"], "relation": "place+food"},
    {"surface": "OxfordSoup", "components": ["Oxford", "Soup"], "relation": "place+food"},
    {"surface": "BostonSalad", "components": ["Boston", "Salad"], "relation": "place+food"},
    {"surface": "PraguePancake", "components": ["Prague", "Pancake"], "relation": "place+food"},
    {"surface": "LimaCookie", "components": ["Lima", "Cookie"], "relation": "place+food"},
    {"surface": "MumbaiWaffle", "components": ["Mumbai", "Waffle"], "relation": "place+food"},
    {"surface": "BangkokMuffin", "components": ["Bangkok", "Muffin"], "relation": "place+food"},
    {"surface": "HanoiDonut", "components": ["Hanoi", "Donut"], "relation": "place+food"},
    {"surface": "LisbonSteak", "components": ["Lisbon", "Steak"], "relation": "place+food"},
    {"surface": "MilanTaco", "components": ["Milan", "Taco"], "relation": "place+food"},
    {"surface": "GenevaBurger", "components": ["Geneva", "Burger"], "relation": "place+food"},
    {"surface": "FlorencePizza", "components": ["Florence", "Pizza"], "relation": "place+food"},
]

VOCAB_PLACE_STRUCTURE = [
    {"surface": "TokyoBridge", "components": ["Tokyo", "Bridge"], "relation": "place+structure"},
    {"surface": "SeoulTower", "components": ["Seoul", "Tower"], "relation": "place+structure"},
    {"surface": "CairoTemple", "components": ["Cairo", "Temple"], "relation": "place+structure"},
    {"surface": "SydneyArch", "components": ["Sydney", "Arch"], "relation": "place+structure"},
    {"surface": "AthensFountain", "components": ["Athens", "Fountain"], "relation": "place+structure"},
    {"surface": "DublinCastle", "components": ["Dublin", "Castle"], "relation": "place+structure"},
    {"surface": "MoscowDome", "components": ["Moscow", "Dome"], "relation": "place+structure"},
    {"surface": "IstanbulGate", "components": ["Istanbul", "Gate"], "relation": "place+structure"},
    {"surface": "PragueStatue", "components": ["Prague", "Statue"], "relation": "place+structure"},
    {"surface": "LimaMonument", "components": ["Lima", "Monument"], "relation": "place+structure"},
    {"surface": "MumbaiFortress", "components": ["Mumbai", "Fortress"], "relation": "place+structure"},
    {"surface": "BangkokPalace", "components": ["Bangkok", "Palace"], "relation": "place+structure"},
    {"surface": "HanoiLighthouse", "components": ["Hanoi", "Lighthouse"], "relation": "place+structure"},
    {"surface": "LisbonPier", "components": ["Lisbon", "Pier"], "relation": "place+structure"},
    {"surface": "MilanPillar", "components": ["Milan", "Pillar"], "relation": "place+structure"},
    {"surface": "GenevaWall", "components": ["Geneva", "Wall"], "relation": "place+structure"},
    {"surface": "FlorenceTunnel", "components": ["Florence", "Tunnel"], "relation": "place+structure"},
    {"surface": "VeniceCanal", "components": ["Venice", "Canal"], "relation": "place+structure"},
    {"surface": "NaplesDock", "components": ["Naples", "Dock"], "relation": "place+structure"},
    {"surface": "LyonAqueduct", "components": ["Lyon", "Aqueduct"], "relation": "place+structure"},
    {"surface": "HamburgHarbor", "components": ["Hamburg", "Harbor"], "relation": "place+structure"},
]

VOCAB_PLACE_INSTITUTION = [
    {"surface": "BostonAcademy", "components": ["Boston", "Academy"], "relation": "place+institution"},
    {"surface": "ViennaOrchestra", "components": ["Vienna", "Orchestra"], "relation": "place+institution"},
    {"surface": "OxfordLibrary", "components": ["Oxford", "Library"], "relation": "place+institution"},
    {"surface": "CairoMuseum", "components": ["Cairo", "Museum"], "relation": "place+institution"},
    {"surface": "SeoulInstitute", "components": ["Seoul", "Institute"], "relation": "place+institution"},
    {"surface": "DublinTheater", "components": ["Dublin", "Theater"], "relation": "place+institution"},
    {"surface": "MoscowGallery", "components": ["Moscow", "Gallery"], "relation": "place+institution"},
    {"surface": "PragueObservatory", "components": ["Prague", "Observatory"], "relation": "place+institution"},
    {"surface": "LimaHospital", "components": ["Lima", "Hospital"], "relation": "place+institution"},
    {"surface": "MumbaiUniversity", "components": ["Mumbai", "University"], "relation": "place+institution"},
    {"surface": "BangkokSeminary", "components": ["Bangkok", "Seminary"], "relation": "place+institution"},
    {"surface": "HanoiConservatory", "components": ["Hanoi", "Conservatory"], "relation": "place+institution"},
    {"surface": "LisbonArchive", "components": ["Lisbon", "Archive"], "relation": "place+institution"},
    {"surface": "MilanFoundation", "components": ["Milan", "Foundation"], "relation": "place+institution"},
    {"surface": "GenevaForum", "components": ["Geneva", "Forum"], "relation": "place+institution"},
]

VOCAB_MATERIAL_ARTIFACT = [
    {"surface": "JadeSculpture", "components": ["Jade", "Sculpture"], "relation": "material+artifact"},
    {"surface": "SilverBracelet", "components": ["Silver", "Bracelet"], "relation": "material+artifact"},
    {"surface": "CopperVase", "components": ["Copper", "Vase"], "relation": "material+artifact"},
    {"surface": "GoldRing", "components": ["Gold", "Ring"], "relation": "material+artifact"},
    {"surface": "IronGate", "components": ["Iron", "Gate"], "relation": "material+artifact"},
    {"surface": "BronzeBell", "components": ["Bronze", "Bell"], "relation": "material+artifact"},
    {"surface": "CrystalChandelier", "components": ["Crystal", "Chandelier"], "relation": "material+artifact"},
    {"surface": "MarbleColumn", "components": ["Marble", "Column"], "relation": "material+artifact"},
    {"surface": "GlassWindow", "components": ["Glass", "Window"], "relation": "material+artifact"},
    {"surface": "SilkRobe", "components": ["Silk", "Robe"], "relation": "material+artifact"},
    {"surface": "LeatherBelt", "components": ["Leather", "Belt"], "relation": "material+artifact"},
    {"surface": "WoolBlanket", "components": ["Wool", "Blanket"], "relation": "material+artifact"},
    {"surface": "ClayPot", "components": ["Clay", "Pot"], "relation": "material+artifact"},
    {"surface": "StonePillar", "components": ["Stone", "Pillar"], "relation": "material+artifact"},
    {"surface": "RubberSeal", "components": ["Rubber", "Seal"], "relation": "material+artifact"},
    {"surface": "PearlNecklace", "components": ["Pearl", "Necklace"], "relation": "material+artifact"},
    {"surface": "EbonyChest", "components": ["Ebony", "Chest"], "relation": "material+artifact"},
    {"surface": "BrassCompass", "components": ["Brass", "Compass"], "relation": "material+artifact"},
    {"surface": "TinWhistle", "components": ["Tin", "Whistle"], "relation": "material+artifact"},
    {"surface": "LinenCurtain", "components": ["Linen", "Curtain"], "relation": "material+artifact"},
]

VOCAB_PURPOSE_CONTAINER = [
    {"surface": "SpiceJar", "components": ["Spice", "Jar"], "relation": "purpose+container"},
    {"surface": "WineBarrel", "components": ["Wine", "Barrel"], "relation": "purpose+container"},
    {"surface": "InkBottle", "components": ["Ink", "Bottle"], "relation": "purpose+container"},
    {"surface": "GrainSilo", "components": ["Grain", "Silo"], "relation": "purpose+container"},
    {"surface": "WaterTank", "components": ["Water", "Tank"], "relation": "purpose+container"},
    {"surface": "OilLamp", "components": ["Oil", "Lamp"], "relation": "purpose+container"},
    {"surface": "TeaPot", "components": ["Tea", "Pot"], "relation": "purpose+container"},
    {"surface": "CoalBin", "components": ["Coal", "Bin"], "relation": "purpose+container"},
    {"surface": "FlourBag", "components": ["Flour", "Bag"], "relation": "purpose+container"},
    {"surface": "SeedPouch", "components": ["Seed", "Pouch"], "relation": "purpose+container"},
]

ALL_VOCAB = (
    VOCAB_PLACE_FOOD + VOCAB_PLACE_STRUCTURE + VOCAB_PLACE_INSTITUTION
    + VOCAB_MATERIAL_ARTIFACT + VOCAB_PURPOSE_CONTAINER
)

EVAL_TEMPLATES = {
    "place+food": [
        "A famous local {comp1} from {comp0} is called",
        "In {comp0}, a popular {comp1} is",
        "The {comp1} that {comp0} is known for is",
    ],
    "place+structure": [
        "A famous {comp1} in {comp0} is called",
        "The iconic {comp1} of {comp0} is",
        "In {comp0}, a well-known {comp1} is",
    ],
    "place+institution": [
        "The {comp1} in {comp0} known for excellence is",
        "A famous {comp1} from {comp0} is",
        "In {comp0}, the prominent {comp1} is",
    ],
    "material+artifact": [
        "A famous {comp1} made of {comp0} is called",
        "The {comp1} crafted from {comp0} is known as",
        "A renowned {comp0} {comp1} is",
    ],
    "purpose+container": [
        "A {comp1} designed for storing {comp0} is called",
        "The {comp1} used for {comp0} is known as",
        "A specialized {comp0} {comp1} is",
    ],
}


def get_final_layer_norm(model):
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.final_layer_norm
    if hasattr(model, "transformer"):
        return model.transformer.ln_f
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    return None


@torch.no_grad()
def extract_hidden(model, tokenizer, text, norm_layer):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    outputs = model(input_ids=input_ids, output_hidden_states=True)
    h = outputs.hidden_states[-1][0, -1, :]
    if norm_layer is not None:
        h = norm_layer(h.unsqueeze(0)).squeeze(0)
    return h


@torch.no_grad()
def build_contrastive_heads(model, tokenizer, registry, alphas):
    """対照 CWE ヘッドを構築する。

    各 α に対して:
      W_v = h_pos - α · h_neg
    """
    device = next(model.parameters()).device
    output_weight = model.get_output_embeddings().weight.detach()
    norm_layer = get_final_layer_norm(model)
    all_relations = list(CONTEXT_TEMPLATES.keys())

    # 全仮想トークン × 全カテゴリテンプレートの隠れ状態を事前計算
    hidden_by_token_rel = {}
    surfaces = []

    for vtoken in registry:
        phrase = " ".join(vtoken.components)
        surfaces.append(vtoken.surface)
        own_rel = vtoken.relation.value

        for rel_key, templates in CONTEXT_TEMPLATES.items():
            hs = []
            for tmpl in templates:
                text = tmpl.format(phrase=phrase)
                h = extract_hidden(model, tokenizer, text, norm_layer)
                hs.append(h)
            h_avg = torch.stack(hs).mean(dim=0)
            hidden_by_token_rel[(vtoken.surface, rel_key)] = h_avg

    # 各 α で出力ヘッドを構築
    heads = {}

    for alpha in alphas:
        rows = []
        for vtoken in registry:
            own_rel = vtoken.relation.value
            h_pos = hidden_by_token_rel[(vtoken.surface, own_rel)]

            other_rels = [r for r in all_relations if r != own_rel]
            h_negs = [
                hidden_by_token_rel[(vtoken.surface, r)]
                for r in other_rels
            ]
            h_neg = torch.stack(h_negs).mean(dim=0)

            w = h_pos - alpha * h_neg
            w = _norm_match_to_vocab(w, output_weight)
            rows.append(w)

        W = torch.stack(rows).to(device)
        head = ContextualOutputHead(W, surfaces)
        head_cal = calibrate_output_head(model, tokenizer, head)
        heads[f"α={alpha:.2f}"] = head_cal

    return heads


@torch.no_grad()
def evaluate(model, tokenizer, heads, vocab_entries):
    device = next(model.parameters()).device
    results = defaultdict(list)

    for entry in vocab_entries:
        rel = entry["relation"]
        templates = EVAL_TEMPLATES.get(rel, [])
        comp0 = entry["components"][0]
        comp1 = entry["components"][1]

        for tmpl in templates:
            prompt = tmpl.format(comp0=comp0, comp1=comp1)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            hidden_state = outputs.hidden_states[-1]
            vocab_logits = outputs.logits

            for cond_name, head in heads.items():
                rankings = head.get_virtual_rankings(hidden_state, vocab_logits)
                sorted_r = sorted(rankings, key=lambda r: r["logit"], reverse=True)
                target = next(r for r in rankings if r["surface"] == entry["surface"])

                results[cond_name].append({
                    "word": entry["surface"],
                    "relation": rel,
                    "prompt": prompt,
                    "correct": sorted_r[0]["surface"] == entry["surface"],
                    "target_logit": target["logit"],
                    "top1": sorted_r[0]["surface"],
                    "top1_logit": sorted_r[0]["logit"],
                    "top1_relation": next(
                        (e["relation"] for e in vocab_entries
                         if e["surface"] == sorted_r[0]["surface"]),
                        "unknown",
                    ),
                })

    return results


def print_results(results):
    word_to_rel = {}
    for entries in results.values():
        for e in entries:
            word_to_rel[e["word"]] = e["relation"]

    rel_types = sorted(set(
        e["relation"] for entries in results.values() for e in entries
    ))

    print(f"\n{'='*80}", flush=True)
    print("対照抽出 α 比較（87語規模）", flush=True)
    print(f"{'='*80}\n", flush=True)

    header = f"{'条件':12s} | {'全体':>8s}"
    for rel in rel_types:
        short = rel.split("+")[1][:6]
        header += f" | {short:>8s}"
    header += " | 流出"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for cond_name in sorted(results.keys()):
        entries = results[cond_name]
        total = len(entries)
        correct = sum(1 for e in entries if e["correct"])
        line = f"  {cond_name:10s} | {correct:2d}/{total:3d} ({100*correct/total:4.1f}%)"

        for rel in rel_types:
            rel_entries = [e for e in entries if e["relation"] == rel]
            rel_c = sum(1 for e in rel_entries if e["correct"])
            rel_t = len(rel_entries)
            line += f" | {rel_c:2d}/{rel_t:2d} ({100*rel_c/max(rel_t,1):4.0f}%)"

        wrong = [e for e in entries if not e["correct"]]
        cross = sum(
            1 for e in wrong if word_to_rel.get(e["top1"], "") != e["relation"]
        )
        line += f" | {cross:2d}"
        print(line, flush=True)

    # purpose+container 詳細（最良条件）
    best_cond = max(
        results.keys(),
        key=lambda c: sum(
            1 for e in results[c]
            if e["relation"] == "purpose+container" and e["correct"]
        ),
    )

    print(f"\n--- purpose+container 最良条件: {best_cond} ---", flush=True)
    pur = [e for e in results[best_cond] if e["relation"] == "purpose+container"]
    for e in sorted(pur, key=lambda x: (x["word"], x["prompt"])):
        mark = "○" if e["correct"] else "×"
        margin = e["target_logit"] - e["top1_logit"]
        info = "" if e["correct"] else f" → {e['top1']}({e['top1_relation']})"
        print(
            f"  {mark} {e['word']:14s} 差={margin:+6.2f}{info}"
            f" | {e['prompt'][:45]}",
            flush=True,
        )


def main():
    output_dir = Path("results/contrastive_cwe")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"モデル読み込み: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"デバイス: {device}", flush=True)

    print(f"\n仮想語彙: {len(ALL_VOCAB)}語", flush=True)
    rel_counts = defaultdict(int)
    for entry in ALL_VOCAB:
        rel_counts[entry["relation"]] += 1
    for rel, count in sorted(rel_counts.items()):
        print(f"  {rel}: {count}語", flush=True)

    registry = VocabularyRegistry()
    registry.add_from_dicts(ALL_VOCAB)
    analyzer = TokenizerAnalyzer(tokenizer)
    analyzer.analyze_registry(registry)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"\n対照 CWE ヘッド構築中（α = {alphas}）...", flush=True)
    heads = build_contrastive_heads(model, tokenizer, registry, alphas)
    print(f"条件数: {len(heads)}", flush=True)

    print("\n評価実行中...", flush=True)
    results = evaluate(model, tokenizer, heads, ALL_VOCAB)

    print_results(results)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"results_{timestamp}.json"
    with open(save_path, "w") as f:
        json.dump(dict(results), f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {save_path}", flush=True)


if __name__ == "__main__":
    main()
