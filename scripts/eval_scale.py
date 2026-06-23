"""仮想語彙の規模拡大実験。

仮想語彙を 10, 20, 50, 75, 100 語と段階的に増やし、
仮想語彙内正解率の推移を検証する。
各規模では全関係型から均等にサンプリングし、
multi_context ヘッドと静的合成（z_norm / word_mean）の2条件を比較する。

実行例::

    CUDA_VISIBLE_DEVICES=5 nohup uv run python scripts/eval_scale.py \
        > logs/eval_scale.log 2>&1 &
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
    calibrate_output_head,
)
from vocabsynth.logit_head import (
    AggregationMethod,
    build_component_logit_head,
    compute_logit_statistics,
    BASELINE_PROMPTS,
)
from vocabsynth.registry import RelationType, VocabularyRegistry

MODEL_NAME = "EleutherAI/pythia-410m"

# ====================================================================
# 100語の仮想語彙定義（5つの関係型）
# ====================================================================

VOCAB_PLACE_FOOD: list[dict] = [
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
    {"surface": "RomePasta", "components": ["Rome", "Pasta"],
     "relation": "place+food"},
    {"surface": "SydneyCurry", "components": ["Sydney", "Curry"],
     "relation": "place+food"},
    {"surface": "CairoDumpling", "components": ["Cairo", "Dumpling"],
     "relation": "place+food"},
    {"surface": "SeoulBread", "components": ["Seoul", "Bread"],
     "relation": "place+food"},
    {"surface": "ViennaCheese", "components": ["Vienna", "Cheese"],
     "relation": "place+food"},
    {"surface": "OxfordSoup", "components": ["Oxford", "Soup"],
     "relation": "place+food"},
    {"surface": "BostonSalad", "components": ["Boston", "Salad"],
     "relation": "place+food"},
    {"surface": "PraguePancake", "components": ["Prague", "Pancake"],
     "relation": "place+food"},
    {"surface": "LimaCookie", "components": ["Lima", "Cookie"],
     "relation": "place+food"},
    {"surface": "MumbaiWaffle", "components": ["Mumbai", "Waffle"],
     "relation": "place+food"},
    {"surface": "BangkokMuffin", "components": ["Bangkok", "Muffin"],
     "relation": "place+food"},
    {"surface": "HanoiDonut", "components": ["Hanoi", "Donut"],
     "relation": "place+food"},
    {"surface": "LisbonSteak", "components": ["Lisbon", "Steak"],
     "relation": "place+food"},
    {"surface": "MilanTaco", "components": ["Milan", "Taco"],
     "relation": "place+food"},
    {"surface": "GenevaBurger", "components": ["Geneva", "Burger"],
     "relation": "place+food"},
    {"surface": "FlorencePizza", "components": ["Florence", "Pizza"],
     "relation": "place+food"},
    {"surface": "VeniceSandwich", "components": ["Venice", "Sandwich"],
     "relation": "place+food"},
    {"surface": "NaplesFudge", "components": ["Naples", "Fudge"],
     "relation": "place+food"},
    {"surface": "LyonCustard", "components": ["Lyon", "Custard"],
     "relation": "place+food"},
    {"surface": "HamburgTruffle", "components": ["Hamburg", "Truffle"],
     "relation": "place+food"},
    {"surface": "DublinPorridge", "components": ["Dublin", "Porridge"],
     "relation": "place+food"},
    {"surface": "WarsawCrepe", "components": ["Warsaw", "Crepe"],
     "relation": "place+food"},
    {"surface": "BudapestStrudel", "components": ["Budapest", "Strudel"],
     "relation": "place+food"},
    {"surface": "AnkaraPilaf", "components": ["Ankara", "Pilaf"],
     "relation": "place+food"},
    {"surface": "TehranSorbet", "components": ["Tehran", "Sorbet"],
     "relation": "place+food"},
]

VOCAB_PLACE_STRUCTURE: list[dict] = [
    {"surface": "TokyoBridge", "components": ["Tokyo", "Bridge"],
     "relation": "place+structure"},
    {"surface": "SeoulTower", "components": ["Seoul", "Tower"],
     "relation": "place+structure"},
    {"surface": "CairoTemple", "components": ["Cairo", "Temple"],
     "relation": "place+structure"},
    {"surface": "SydneyArch", "components": ["Sydney", "Arch"],
     "relation": "place+structure"},
    {"surface": "AthensFountain", "components": ["Athens", "Fountain"],
     "relation": "place+structure"},
    {"surface": "DublinCastle", "components": ["Dublin", "Castle"],
     "relation": "place+structure"},
    {"surface": "MoscowDome", "components": ["Moscow", "Dome"],
     "relation": "place+structure"},
    {"surface": "IstanbulGate", "components": ["Istanbul", "Gate"],
     "relation": "place+structure"},
    {"surface": "PragueStatue", "components": ["Prague", "Statue"],
     "relation": "place+structure"},
    {"surface": "LimaMonument", "components": ["Lima", "Monument"],
     "relation": "place+structure"},
    {"surface": "MumbaiFortress", "components": ["Mumbai", "Fortress"],
     "relation": "place+structure"},
    {"surface": "BangkokPalace", "components": ["Bangkok", "Palace"],
     "relation": "place+structure"},
    {"surface": "HanoiLighthouse", "components": ["Hanoi", "Lighthouse"],
     "relation": "place+structure"},
    {"surface": "LisbonPier", "components": ["Lisbon", "Pier"],
     "relation": "place+structure"},
    {"surface": "MilanPillar", "components": ["Milan", "Pillar"],
     "relation": "place+structure"},
    {"surface": "GenevaWall", "components": ["Geneva", "Wall"],
     "relation": "place+structure"},
    {"surface": "FlorenceTunnel", "components": ["Florence", "Tunnel"],
     "relation": "place+structure"},
    {"surface": "VeniceCanal", "components": ["Venice", "Canal"],
     "relation": "place+structure"},
    {"surface": "NaplesDock", "components": ["Naples", "Dock"],
     "relation": "place+structure"},
    {"surface": "LyonAqueduct", "components": ["Lyon", "Aqueduct"],
     "relation": "place+structure"},
    {"surface": "HamburgHarbor", "components": ["Hamburg", "Harbor"],
     "relation": "place+structure"},
    {"surface": "DresdenChapel", "components": ["Dresden", "Chapel"],
     "relation": "place+structure"},
    {"surface": "ZurichClock", "components": ["Zurich", "Clock"],
     "relation": "place+structure"},
    {"surface": "HelsinkiSteeple", "components": ["Helsinki", "Steeple"],
     "relation": "place+structure"},
    {"surface": "OsloPavilion", "components": ["Oslo", "Pavilion"],
     "relation": "place+structure"},
]

VOCAB_PLACE_INSTITUTION: list[dict] = [
    {"surface": "BostonAcademy", "components": ["Boston", "Academy"],
     "relation": "place+institution"},
    {"surface": "ViennaOrchestra", "components": ["Vienna", "Orchestra"],
     "relation": "place+institution"},
    {"surface": "OxfordLibrary", "components": ["Oxford", "Library"],
     "relation": "place+institution"},
    {"surface": "CairoMuseum", "components": ["Cairo", "Museum"],
     "relation": "place+institution"},
    {"surface": "SeoulInstitute", "components": ["Seoul", "Institute"],
     "relation": "place+institution"},
    {"surface": "DublinTheater", "components": ["Dublin", "Theater"],
     "relation": "place+institution"},
    {"surface": "MoscowGallery", "components": ["Moscow", "Gallery"],
     "relation": "place+institution"},
    {"surface": "PragueObservatory", "components": ["Prague", "Observatory"],
     "relation": "place+institution"},
    {"surface": "LimaHospital", "components": ["Lima", "Hospital"],
     "relation": "place+institution"},
    {"surface": "MumbaiUniversity", "components": ["Mumbai", "University"],
     "relation": "place+institution"},
    {"surface": "BangkokSeminary", "components": ["Bangkok", "Seminary"],
     "relation": "place+institution"},
    {"surface": "HanoiConservatory", "components": ["Hanoi", "Conservatory"],
     "relation": "place+institution"},
    {"surface": "LisbonArchive", "components": ["Lisbon", "Archive"],
     "relation": "place+institution"},
    {"surface": "MilanFoundation", "components": ["Milan", "Foundation"],
     "relation": "place+institution"},
    {"surface": "GenevaForum", "components": ["Geneva", "Forum"],
     "relation": "place+institution"},
]

VOCAB_MATERIAL_ARTIFACT: list[dict] = [
    {"surface": "JadeSculpture", "components": ["Jade", "Sculpture"],
     "relation": "material+artifact"},
    {"surface": "SilverBracelet", "components": ["Silver", "Bracelet"],
     "relation": "material+artifact"},
    {"surface": "CopperVase", "components": ["Copper", "Vase"],
     "relation": "material+artifact"},
    {"surface": "GoldRing", "components": ["Gold", "Ring"],
     "relation": "material+artifact"},
    {"surface": "IronGate", "components": ["Iron", "Gate"],
     "relation": "material+artifact"},
    {"surface": "BronzeBell", "components": ["Bronze", "Bell"],
     "relation": "material+artifact"},
    {"surface": "CrystalChandelier", "components": ["Crystal", "Chandelier"],
     "relation": "material+artifact"},
    {"surface": "MarbleColumn", "components": ["Marble", "Column"],
     "relation": "material+artifact"},
    {"surface": "GlassWindow", "components": ["Glass", "Window"],
     "relation": "material+artifact"},
    {"surface": "SilkRobe", "components": ["Silk", "Robe"],
     "relation": "material+artifact"},
    {"surface": "LeatherBelt", "components": ["Leather", "Belt"],
     "relation": "material+artifact"},
    {"surface": "WoolBlanket", "components": ["Wool", "Blanket"],
     "relation": "material+artifact"},
    {"surface": "ClayPot", "components": ["Clay", "Pot"],
     "relation": "material+artifact"},
    {"surface": "StonePillar", "components": ["Stone", "Pillar"],
     "relation": "material+artifact"},
    {"surface": "RubberSeal", "components": ["Rubber", "Seal"],
     "relation": "material+artifact"},
    {"surface": "PearlNecklace", "components": ["Pearl", "Necklace"],
     "relation": "material+artifact"},
    {"surface": "EbonyChest", "components": ["Ebony", "Chest"],
     "relation": "material+artifact"},
    {"surface": "BrassCompass", "components": ["Brass", "Compass"],
     "relation": "material+artifact"},
    {"surface": "TinWhistle", "components": ["Tin", "Whistle"],
     "relation": "material+artifact"},
    {"surface": "LinenCurtain", "components": ["Linen", "Curtain"],
     "relation": "material+artifact"},
]

VOCAB_PURPOSE_CONTAINER: list[dict] = [
    {"surface": "SpiceJar", "components": ["Spice", "Jar"],
     "relation": "purpose+container"},
    {"surface": "WineBarrel", "components": ["Wine", "Barrel"],
     "relation": "purpose+container"},
    {"surface": "InkBottle", "components": ["Ink", "Bottle"],
     "relation": "purpose+container"},
    {"surface": "GrainSilo", "components": ["Grain", "Silo"],
     "relation": "purpose+container"},
    {"surface": "WaterTank", "components": ["Water", "Tank"],
     "relation": "purpose+container"},
    {"surface": "OilLamp", "components": ["Oil", "Lamp"],
     "relation": "purpose+container"},
    {"surface": "TeaPot", "components": ["Tea", "Pot"],
     "relation": "purpose+container"},
    {"surface": "CoalBin", "components": ["Coal", "Bin"],
     "relation": "purpose+container"},
    {"surface": "FlourBag", "components": ["Flour", "Bag"],
     "relation": "purpose+container"},
    {"surface": "SeedPouch", "components": ["Seed", "Pouch"],
     "relation": "purpose+container"},
]

# 全関係型のリスト（順序固定）
ALL_RELATION_POOLS: list[tuple[str, list[dict]]] = [
    ("place+food", VOCAB_PLACE_FOOD),
    ("place+structure", VOCAB_PLACE_STRUCTURE),
    ("place+institution", VOCAB_PLACE_INSTITUTION),
    ("material+artifact", VOCAB_MATERIAL_ARTIFACT),
    ("purpose+container", VOCAB_PURPOSE_CONTAINER),
]

# ====================================================================
# 各関係型の評価プロンプトテンプレート（3つずつ）
# ====================================================================

EVAL_TEMPLATES: dict[str, list[dict]] = {
    "place+food": [
        {"template": "A famous local {comp1} from {comp0} is called",
         "description": "地名先行+食物"},
        {"template": "In {comp0}, a popular {comp1} is",
         "description": "場所導入+食物"},
        {"template": "The {comp1} that {comp0} is known for is",
         "description": "食物先行+地名"},
    ],
    "place+structure": [
        {"template": "A famous {comp1} in {comp0} is called",
         "description": "構造物+地名"},
        {"template": "The iconic {comp1} of {comp0} is",
         "description": "構造物先行"},
        {"template": "In {comp0}, a well-known {comp1} is",
         "description": "場所導入+構造物"},
    ],
    "place+institution": [
        {"template": "The {comp1} in {comp0} known for excellence is",
         "description": "機関+地名"},
        {"template": "A famous {comp1} from {comp0} is",
         "description": "機関先行+地名"},
        {"template": "In {comp0}, the prominent {comp1} is",
         "description": "場所導入+機関"},
    ],
    "material+artifact": [
        {"template": "A famous {comp1} made of {comp0} is called",
         "description": "工芸品+素材"},
        {"template": "The {comp1} crafted from {comp0} is known as",
         "description": "工芸品先行+素材"},
        {"template": "A renowned {comp0} {comp1} is",
         "description": "素材+工芸品 直接連結"},
    ],
    "purpose+container": [
        {"template": "A {comp1} designed for storing {comp0} is called",
         "description": "容器+用途"},
        {"template": "The {comp1} used for {comp0} is known as",
         "description": "容器先行+用途"},
        {"template": "A specialized {comp0} {comp1} is",
         "description": "用途+容器 直接連結"},
    ],
}


def select_vocab_for_scale(scale: int) -> list[dict]:
    """指定規模の仮想語彙を各関係型から均等にサンプリングする。

    各関係型のプール上限を超えない範囲で均等に割り当てる。
    """
    n_types = len(ALL_RELATION_POOLS)
    # まず均等に割り当てる基本数
    base_per_type = scale // n_types
    remainder = scale % n_types

    selected: list[dict] = []
    for i, (_, pool) in enumerate(ALL_RELATION_POOLS):
        n = base_per_type + (1 if i < remainder else 0)
        n = min(n, len(pool))
        selected.extend(pool[:n])

    # 均等割り当てで足りなかった分を大きいプールから補充
    while len(selected) < scale:
        for _, pool in ALL_RELATION_POOLS:
            already = sum(
                1 for s in selected
                if s["relation"] == pool[0]["relation"]
            )
            if already < len(pool):
                selected.append(pool[already])
                if len(selected) >= scale:
                    break
        else:
            # 全プールを使い切った場合は終了
            break

    return selected[:scale]


def build_eval_prompts(
    vocab_entries: list[dict],
) -> list[dict]:
    """仮想語彙エントリから評価プロンプト一覧を生成する。"""
    prompts = []
    for entry in vocab_entries:
        rel = entry["relation"]
        templates = EVAL_TEMPLATES.get(rel, [])
        comp0 = entry["components"][0]
        comp1 = entry["components"][1]
        for tmpl_info in templates:
            prompt_text = tmpl_info["template"].format(
                comp0=comp0, comp1=comp1,
            )
            prompts.append({
                "prompt": prompt_text,
                "expected": entry["surface"],
                "relation": rel,
            })
    return prompts


@torch.no_grad()
def evaluate_within_virtual_accuracy(
    model,
    tokenizer,
    head,
    prompts: list[dict],
    is_component_head: bool = False,
) -> tuple[float, list[dict]]:
    """仮想語彙内正解率を計算する。

    対象仮想語彙がN語中で最大ロジットとなる割合を返す。
    """
    device = next(model.parameters()).device
    hit = 0
    total = len(prompts)
    details = []

    for item in prompts:
        prompt = item["prompt"]
        expected = item["expected"]

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1]
        vocab_logits = outputs.logits

        rankings = head.get_virtual_rankings(hidden_state, vocab_logits)

        # 仮想語彙内での順位
        # ロジット降順でソート
        sorted_by_logit = sorted(
            rankings, key=lambda r: r["logit"], reverse=True,
        )
        rank_in_virtual = -1
        for vi, r in enumerate(sorted_by_logit):
            if r["surface"] == expected:
                rank_in_virtual = vi
                break

        is_top1 = rank_in_virtual == 0
        if is_top1:
            hit += 1

        details.append({
            "prompt": prompt,
            "expected": expected,
            "relation": item["relation"],
            "rank_in_virtual": rank_in_virtual,
            "is_top1": is_top1,
            "expected_logit": next(
                r["logit"] for r in rankings if r["surface"] == expected
            ),
            "top1_surface": sorted_by_logit[0]["surface"],
            "top1_logit": sorted_by_logit[0]["logit"],
        })

    accuracy = hit / total if total > 0 else 0.0
    return accuracy, details


def main() -> None:
    output_dir = Path("results/scale")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("仮想語彙 規模拡大実験", flush=True)
    print("=" * 70, flush=True)

    print(f"\nモデル読み込み: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"デバイス: {device}", flush=True)

    # ロジット統計を事前計算（静的合成の z_norm 用）
    print("\nロジット統計を事前計算中...", flush=True)
    baseline_mean, baseline_std = compute_logit_statistics(
        model, tokenizer, BASELINE_PROMPTS,
    )
    print("  完了", flush=True)

    # ========================================
    # 規模別の評価
    # ========================================
    scales = [10, 20, 50, 75, 100]
    all_scale_results = []

    for scale in scales:
        print(f"\n{'=' * 70}", flush=True)
        print(f"規模: {scale}語", flush=True)
        print(f"{'=' * 70}", flush=True)

        # 仮想語彙の選択
        vocab_entries = select_vocab_for_scale(scale)
        actual_scale = len(vocab_entries)
        print(f"  実際の語彙数: {actual_scale}", flush=True)

        # 関係型ごとの内訳を表示
        rel_counts: dict[str, int] = {}
        for entry in vocab_entries:
            rel = entry["relation"]
            rel_counts[rel] = rel_counts.get(rel, 0) + 1
        for rel, count in sorted(rel_counts.items()):
            print(f"    {rel}: {count}語", flush=True)

        # レジストリ構築
        registry = VocabularyRegistry()
        registry.add_from_dicts(vocab_entries)
        analyzer = TokenizerAnalyzer(tokenizer)
        analyzer.analyze_registry(registry)

        # 評価プロンプト生成
        eval_prompts = build_eval_prompts(vocab_entries)
        print(f"  評価プロンプト数: {len(eval_prompts)}", flush=True)

        scale_result: dict = {
            "scale": actual_scale,
            "relation_counts": rel_counts,
            "num_prompts": len(eval_prompts),
            "conditions": {},
        }

        # --------------------------------------------------
        # 条件1: multi_context ヘッド
        # --------------------------------------------------
        print(f"\n  --- 条件: multi_context ---", flush=True)
        head_raw = build_contextual_output_head(
            model, tokenizer, registry,
            method=ContextualMethod.MULTI_CONTEXT,
        )
        head_mc = calibrate_output_head(model, tokenizer, head_raw)
        cal_info = head_mc.layer_info[-1]
        print(
            f"    キャリブレーション: "
            f"スケール={cal_info['calibration_scale']:.4f}, "
            f"仮想平均={cal_info['avg_virtual_before']:.2f}, "
            f"語彙上位50={cal_info['avg_vocab_top50']:.2f}",
            flush=True,
        )

        acc_mc, details_mc = evaluate_within_virtual_accuracy(
            model, tokenizer, head_mc, eval_prompts,
        )
        print(
            f"    仮想語彙内正解率: {acc_mc * 100:.1f}% "
            f"({sum(1 for d in details_mc if d['is_top1'])}"
            f"/{len(details_mc)})",
            flush=True,
        )

        # 関係型別の正解率
        rel_acc_mc: dict[str, dict] = {}
        for d in details_mc:
            rel = d["relation"]
            if rel not in rel_acc_mc:
                rel_acc_mc[rel] = {"hit": 0, "total": 0}
            rel_acc_mc[rel]["total"] += 1
            if d["is_top1"]:
                rel_acc_mc[rel]["hit"] += 1
        for rel, stats in sorted(rel_acc_mc.items()):
            rate = stats["hit"] / stats["total"] * 100
            print(f"      {rel}: {rate:.1f}%", flush=True)

        scale_result["conditions"]["multi_context"] = {
            "accuracy": acc_mc,
            "relation_accuracy": {
                k: v["hit"] / v["total"]
                for k, v in rel_acc_mc.items()
            },
            "calibration": cal_info,
            "details": details_mc,
        }

        # --------------------------------------------------
        # 条件2: 静的合成 z_norm (word_mean)
        # --------------------------------------------------
        print(f"\n  --- 条件: z_norm / word_mean ---", flush=True)
        head_znorm = build_component_logit_head(
            tokenizer, registry,
            aggregation=AggregationMethod.WORD_MEAN,
            baseline=baseline_mean,
            baseline_std=baseline_std,
        )

        acc_zn, details_zn = evaluate_within_virtual_accuracy(
            model, tokenizer, head_znorm, eval_prompts,
            is_component_head=True,
        )
        print(
            f"    仮想語彙内正解率: {acc_zn * 100:.1f}% "
            f"({sum(1 for d in details_zn if d['is_top1'])}"
            f"/{len(details_zn)})",
            flush=True,
        )

        rel_acc_zn: dict[str, dict] = {}
        for d in details_zn:
            rel = d["relation"]
            if rel not in rel_acc_zn:
                rel_acc_zn[rel] = {"hit": 0, "total": 0}
            rel_acc_zn[rel]["total"] += 1
            if d["is_top1"]:
                rel_acc_zn[rel]["hit"] += 1
        for rel, stats in sorted(rel_acc_zn.items()):
            rate = stats["hit"] / stats["total"] * 100
            print(f"      {rel}: {rate:.1f}%", flush=True)

        scale_result["conditions"]["z_norm_word_mean"] = {
            "accuracy": acc_zn,
            "relation_accuracy": {
                k: v["hit"] / v["total"]
                for k, v in rel_acc_zn.items()
            },
            "details": details_zn,
        }

        all_scale_results.append(scale_result)

    # ========================================
    # 正解率推移の集約表示
    # ========================================
    print(f"\n\n{'=' * 70}", flush=True)
    print("正解率推移（規模 vs 正解率）", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(
        f"{'規模':>6s} | {'multi_context':>14s} | {'z_norm/word_mean':>16s}",
        flush=True,
    )
    print("-" * 44, flush=True)

    for sr in all_scale_results:
        s = sr["scale"]
        mc = sr["conditions"]["multi_context"]["accuracy"] * 100
        zn = sr["conditions"]["z_norm_word_mean"]["accuracy"] * 100
        print(f"{s:6d} | {mc:13.1f}% | {zn:15.1f}%", flush=True)

    # 関係型別の推移
    print(f"\n\n{'=' * 70}", flush=True)
    print("関係型別 multi_context 正解率の推移", flush=True)
    print(f"{'=' * 70}", flush=True)

    all_relations = sorted(set(
        rel
        for sr in all_scale_results
        for rel in sr["conditions"]["multi_context"]["relation_accuracy"]
    ))

    header = f"{'規模':>6s}"
    for rel in all_relations:
        header += f" | {rel:>20s}"
    print(header, flush=True)
    print("-" * (8 + 23 * len(all_relations)), flush=True)

    for sr in all_scale_results:
        s = sr["scale"]
        row = f"{s:6d}"
        rel_acc = sr["conditions"]["multi_context"]["relation_accuracy"]
        for rel in all_relations:
            rate = rel_acc.get(rel, 0.0) * 100
            row += f" | {rate:19.1f}%"
        print(row, flush=True)

    # ========================================
    # 結果保存
    # ========================================
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"

    # details を保存用に圧縮（ファイルサイズ削減）
    save_results = []
    for sr in all_scale_results:
        save_sr = {
            "scale": sr["scale"],
            "relation_counts": sr["relation_counts"],
            "num_prompts": sr["num_prompts"],
            "conditions": {},
        }
        for cond_name, cond_data in sr["conditions"].items():
            save_sr["conditions"][cond_name] = {
                "accuracy": cond_data["accuracy"],
                "relation_accuracy": cond_data.get("relation_accuracy", {}),
            }
            if "calibration" in cond_data:
                save_sr["conditions"][cond_name]["calibration"] = (
                    cond_data["calibration"]
                )
            # 詳細結果も含める（分析用）
            save_sr["conditions"][cond_name]["details"] = cond_data["details"]
        save_results.append(save_sr)

    summary = {
        "model": MODEL_NAME,
        "timestamp": timestamp,
        "scales": scales,
        "total_vocab_pool": 100,
        "scale_results": save_results,
        "accuracy_curve": {
            "scales": [sr["scale"] for sr in all_scale_results],
            "multi_context": [
                sr["conditions"]["multi_context"]["accuracy"]
                for sr in all_scale_results
            ],
            "z_norm_word_mean": [
                sr["conditions"]["z_norm_word_mean"]["accuracy"]
                for sr in all_scale_results
            ],
        },
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)
    print("完了", flush=True)


if __name__ == "__main__":
    main()
