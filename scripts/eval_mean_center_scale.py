"""87語規模での平均中心化の効果検証。

20語規模では平均中心化が88→92%の改善を示した。
87語規模（purpose+container が33%に低下する条件）で同手法が
カテゴリ間流出を抑制するかを検証する。

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_mean_center_scale.py
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
    ContextualMethod,
    ContextualOutputHead,
    build_contextual_output_head,
    calibrate_output_head,
)
from vocabsynth.registry import VocabularyRegistry

MODEL_NAME = "EleutherAI/pythia-410m"

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


def build_mean_centered_head(base_head, model):
    """平均中心化版の出力ヘッドを構築する。"""
    W = base_head._U.clone()
    W_centered = W - W.mean(dim=0, keepdim=True)

    output_weight = model.get_output_embeddings().weight.detach()
    vocab_median_norm = output_weight.norm(dim=1).median()
    norms = W_centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
    W_centered = W_centered * (vocab_median_norm / norms)

    return ContextualOutputHead(W_centered, base_head.surface_names)


@torch.no_grad()
def evaluate(model, tokenizer, head, vocab_entries):
    """仮想語彙内正解率を計算する。"""
    device = next(model.parameters()).device
    results = []

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

            rankings = head.get_virtual_rankings(hidden_state, vocab_logits)
            sorted_rankings = sorted(
                rankings, key=lambda r: r["logit"], reverse=True,
            )

            target = next(r for r in rankings if r["surface"] == entry["surface"])
            top1 = sorted_rankings[0]

            results.append({
                "word": entry["surface"],
                "relation": rel,
                "prompt": prompt,
                "correct": top1["surface"] == entry["surface"],
                "target_logit": target["logit"],
                "top1": top1["surface"],
                "top1_logit": top1["logit"],
                "top1_relation": next(
                    (e["relation"] for e in vocab_entries
                     if e["surface"] == top1["surface"]),
                    "unknown",
                ),
            })

    return results


def print_results(results_by_condition, vocab_entries):
    """結果を表示する。"""
    word_to_rel = {e["surface"]: e["relation"] for e in vocab_entries}
    rel_types = sorted(set(e["relation"] for e in vocab_entries))

    print(f"\n{'='*70}", flush=True)
    print("87語規模 平均中心化実験 結果", flush=True)
    print(f"{'='*70}\n", flush=True)

    for cond_name, results in results_by_condition.items():
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        print(f"--- {cond_name} ---", flush=True)
        print(f"  全体: {correct}/{total} ({100*correct/total:.1f}%)", flush=True)

        for rel in rel_types:
            rel_results = [r for r in results if r["relation"] == rel]
            rel_correct = sum(1 for r in rel_results if r["correct"])
            rel_total = len(rel_results)
            print(
                f"  {rel:25s}: {rel_correct:2d}/{rel_total:2d}"
                f" ({100*rel_correct/max(rel_total,1):.1f}%)",
                flush=True,
            )

        wrong = [r for r in results if not r["correct"]]
        same = sum(1 for r in wrong if r["top1_relation"] == r["relation"])
        cross = len(wrong) - same
        print(f"  誤答: {len(wrong)}件 (同カテゴリ:{same} 他カテゴリ:{cross})", flush=True)
        print(flush=True)

    # purpose+container 詳細比較
    print(f"{'='*70}", flush=True)
    print("purpose+container 詳細比較", flush=True)
    print(f"{'='*70}\n", flush=True)

    for cond_name, results in results_by_condition.items():
        pur = [r for r in results if r["relation"] == "purpose+container"]
        pur_c = sum(1 for r in pur if r["correct"])
        print(f"--- {cond_name}: {pur_c}/{len(pur)} ---", flush=True)
        for r in sorted(pur, key=lambda x: (x["word"], x["prompt"])):
            mark = "○" if r["correct"] else "×"
            margin = r["target_logit"] - r["top1_logit"]
            info = "" if r["correct"] else f" → {r['top1']}({r['top1_relation']})"
            print(
                f"  {mark} {r['word']:14s} 差={margin:+6.2f}{info}"
                f" | {r['prompt'][:45]}",
                flush=True,
            )
        print(flush=True)

    # material+artifact 誤答のみ
    print(f"{'='*70}", flush=True)
    print("material+artifact 誤答詳細", flush=True)
    print(f"{'='*70}\n", flush=True)

    for cond_name, results in results_by_condition.items():
        mat_wrong = [
            r for r in results
            if r["relation"] == "material+artifact" and not r["correct"]
        ]
        print(f"--- {cond_name}: 誤答{len(mat_wrong)}件 ---", flush=True)
        for r in sorted(mat_wrong, key=lambda x: x["word"]):
            print(
                f"  × {r['word']:18s} → {r['top1']:18s}({r['top1_relation']})"
                f" 差={r['target_logit']-r['top1_logit']:+.2f}",
                flush=True,
            )
        print(flush=True)


def main():
    output_dir = Path("results/mean_center_scale")
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

    print("\n出力ヘッド構築中（CWE multi_context）...", flush=True)
    head_raw = build_contextual_output_head(
        model, tokenizer, registry,
        method=ContextualMethod.MULTI_CONTEXT,
    )

    print("キャリブレーション前に平均中心化ヘッドを構築...", flush=True)
    head_centered_raw = build_mean_centered_head(head_raw, model)

    print("キャリブレーション...", flush=True)
    head_baseline = calibrate_output_head(model, tokenizer, head_raw)
    head_centered = calibrate_output_head(model, tokenizer, head_centered_raw)

    cal_bl = head_baseline.layer_info[-1]
    cal_ct = head_centered.layer_info[-1]
    print(
        f"  baseline: スケール={cal_bl['calibration_scale']:.4f},"
        f" 仮想平均={cal_bl['avg_virtual_before']:.2f},"
        f" 語彙上位50={cal_bl['avg_vocab_top50']:.2f}",
        flush=True,
    )
    print(
        f"  mean_center: スケール={cal_ct['calibration_scale']:.4f},"
        f" 仮想平均={cal_ct['avg_virtual_before']:.2f},"
        f" 語彙上位50={cal_ct['avg_vocab_top50']:.2f}",
        flush=True,
    )

    print("\n評価実行中...", flush=True)
    results_baseline = evaluate(model, tokenizer, head_baseline, ALL_VOCAB)
    results_centered = evaluate(model, tokenizer, head_centered, ALL_VOCAB)

    results_by_condition = {
        "multi_context(baseline)": results_baseline,
        "multi_context+mean_center": results_centered,
    }

    print_results(results_by_condition, ALL_VOCAB)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"results_{timestamp}.json"
    with open(save_path, "w") as f:
        json.dump(results_by_condition, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {save_path}", flush=True)


if __name__ == "__main__":
    main()
