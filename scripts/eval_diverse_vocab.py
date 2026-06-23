"""多様な仮想語彙による頑健性検証実験。

5つの意味領域（地名+食物、地名+構造物、地名+機関、
素材+工芸品、用途+容器）にまたがる仮想語彙を構成し、
内部表現抽出型ヘッド + キャリブレーション + 反復抑制の
組み合わせで端到端の生成精度と弁別性を検証する。

検証観点:
  1. 正答率: 文脈に合致する仮想語彙が第一トークンで選ばれるか
  2. 弁別性: 同じ関係タイプの仮想語彙が正しく区別されるか
  3. 抑制性: 無関係な文脈で仮想語彙が出現しないか
  4. 語彙規模の影響: 仮想語彙を6件から20件に拡大しても精度が維持されるか

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_diverse_vocab.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
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

# ========================================
# 多様な仮想語彙の定義（5つの関係タイプ、合計20件）
# ========================================

DIVERSE_VIRTUAL_TOKENS = [
    # --- 地名 + 食物 (place+food): 6件 ---
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

    # --- 地名 + 構造物 (place+structure): 4件 ---
    {"surface": "TokyoBridge", "components": ["Tokyo", "Bridge"],
     "relation": "place+structure"},
    {"surface": "SeoulTower", "components": ["Seoul", "Tower"],
     "relation": "place+structure"},
    {"surface": "CairoTemple", "components": ["Cairo", "Temple"],
     "relation": "place+structure"},
    {"surface": "SydneyArch", "components": ["Sydney", "Arch"],
     "relation": "place+structure"},

    # --- 地名 + 機関 (place+institution): 3件 ---
    {"surface": "BostonAcademy", "components": ["Boston", "Academy"],
     "relation": "place+institution"},
    {"surface": "ViennaOrchestra", "components": ["Vienna", "Orchestra"],
     "relation": "place+institution"},
    {"surface": "OxfordLibrary", "components": ["Oxford", "Library"],
     "relation": "place+institution"},

    # --- 素材 + 工芸品 (material+artifact): 4件 ---
    {"surface": "JadeSculpture", "components": ["Jade", "Sculpture"],
     "relation": "material+artifact"},
    {"surface": "SilverBracelet", "components": ["Silver", "Bracelet"],
     "relation": "material+artifact"},
    {"surface": "CopperVase", "components": ["Copper", "Vase"],
     "relation": "material+artifact"},
    {"surface": "IvoryChess", "components": ["Ivory", "Chess"],
     "relation": "material+artifact"},

    # --- 用途 + 容器 (purpose+container): 3件 ---
    {"surface": "SpiceJar", "components": ["Spice", "Jar"],
     "relation": "purpose+container"},
    {"surface": "WineBarrel", "components": ["Wine", "Barrel"],
     "relation": "purpose+container"},
    {"surface": "InkBottle", "components": ["Ink", "Bottle"],
     "relation": "purpose+container"},
]


# ========================================
# 生成プロンプト: 各仮想語彙に対して2〜3件ずつ
# ========================================

GENERATION_PROMPTS = [
    # --- place+food ---
    {"prompt": "A famous local Cake from Narita is called",
     "expected": "NaritaCake"},
    {"prompt": "In Osaka, a popular Noodle is",
     "expected": "OsakaNoodle"},
    {"prompt": "A famous local Pretzel from Berlin is called",
     "expected": "BerlinPretzel"},
    {"prompt": "In Paris, a popular Chocolate is",
     "expected": "ParisChocolate"},
    {"prompt": "A famous local Pie from London is called",
     "expected": "LondonPie"},
    {"prompt": "The Pasta that Rome is famous for is",
     "expected": "RomePasta"},
    {"prompt": "Tourists visiting Narita try the local Cake known as",
     "expected": "NaritaCake"},
    {"prompt": "The Noodle dish from Osaka called",
     "expected": "OsakaNoodle"},

    # --- place+structure ---
    {"prompt": "A famous Bridge in Tokyo is called",
     "expected": "TokyoBridge"},
    {"prompt": "The Tower in Seoul that tourists visit is",
     "expected": "SeoulTower"},
    {"prompt": "A famous Temple in Cairo is called",
     "expected": "CairoTemple"},
    {"prompt": "The Arch in Sydney that everyone sees is",
     "expected": "SydneyArch"},
    {"prompt": "The iconic Bridge of Tokyo is",
     "expected": "TokyoBridge"},

    # --- place+institution ---
    {"prompt": "The Academy in Boston known for excellence is",
     "expected": "BostonAcademy"},
    {"prompt": "A famous Orchestra from Vienna is",
     "expected": "ViennaOrchestra"},
    {"prompt": "The Library at Oxford that scholars visit is",
     "expected": "OxfordLibrary"},

    # --- material+artifact ---
    {"prompt": "A famous Sculpture made of Jade is called",
     "expected": "JadeSculpture"},
    {"prompt": "The Bracelet crafted from Silver is known as",
     "expected": "SilverBracelet"},
    {"prompt": "A renowned Vase made of Copper is called",
     "expected": "CopperVase"},
    {"prompt": "The Chess set carved from Ivory is known as",
     "expected": "IvoryChess"},

    # --- purpose+container ---
    {"prompt": "A Jar designed for storing Spice is called",
     "expected": "SpiceJar"},
    {"prompt": "The Barrel used for aging Wine is known as",
     "expected": "WineBarrel"},
    {"prompt": "A Bottle designed to hold Ink is called",
     "expected": "InkBottle"},
]


# ========================================
# 弁別性テスト: 紛らわしい対が正しく区別されるか
# ========================================

DISCRIMINATION_PROMPTS = [
    # 同じ地名+食物カテゴリ内での区別
    {"prompt": "In Berlin, the local Pretzel is",
     "expected": "BerlinPretzel",
     "confuser": "BerlinPretzel vs ParisChocolate"},
    {"prompt": "In Paris, the famous Chocolate is",
     "expected": "ParisChocolate",
     "confuser": "ParisChocolate vs BerlinPretzel"},
    # 同じ地名+構造物カテゴリ内での区別
    {"prompt": "The Bridge in Tokyo is",
     "expected": "TokyoBridge",
     "confuser": "TokyoBridge vs SeoulTower"},
    {"prompt": "The Tower in Seoul is",
     "expected": "SeoulTower",
     "confuser": "SeoulTower vs TokyoBridge"},
    # 同じ素材+工芸品カテゴリ内での区別
    {"prompt": "The Sculpture made of Jade is",
     "expected": "JadeSculpture",
     "confuser": "JadeSculpture vs SilverBracelet"},
    {"prompt": "The Bracelet made of Silver is",
     "expected": "SilverBracelet",
     "confuser": "SilverBracelet vs JadeSculpture"},
    # カテゴリ横断での区別
    {"prompt": "The Barrel for Wine is",
     "expected": "WineBarrel",
     "confuser": "WineBarrel vs WineBarrel（容器 vs 食物）"},
    {"prompt": "The Academy in Boston is",
     "expected": "BostonAcademy",
     "confuser": "BostonAcademy vs OxfordLibrary"},
]


# ========================================
# 抑制性テスト: 仮想語彙が出るべきでない文脈
# ========================================

SUPPRESSION_PROMPTS = [
    {"prompt": "The weather forecast for tomorrow says",
     "expected": "__none__"},
    {"prompt": "The mathematical proof begins with",
     "expected": "__none__"},
    {"prompt": "In a galaxy far, far away there was",
     "expected": "__none__"},
    {"prompt": "The stock market opened today with",
     "expected": "__none__"},
    {"prompt": "According to the latest scientific study",
     "expected": "__none__"},
    {"prompt": "The computer program runs by first",
     "expected": "__none__"},
]


@torch.no_grad()
def generate_with_virtual_vocab(
    model,
    tokenizer,
    head: ContextualOutputHead,
    prompt: str,
    max_new_tokens: int = 10,
    greedy: bool = True,
    suppress_virtual_repeat: bool = True,
    virtual_cooldown: int = 1,
) -> dict:
    """仮想語彙を含む自己回帰生成。"""
    device = next(model.parameters()).device
    V = model.config.vocab_size

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    current_ids = input_ids

    generated_tokens = []
    generated_text_parts = []
    used_virtual = set()
    cooldown_remaining = 0

    for step in range(max_new_tokens):
        outputs = model(input_ids=current_ids, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1]
        vocab_logits = outputs.logits

        last_h = hidden_state[:, -1:, :]
        last_vocab_logits = vocab_logits[:, -1:, :]
        extended_logits = head.extend_logits(last_h, last_vocab_logits)
        logits_1d = extended_logits[0, 0, :]

        # 反復抑制
        if suppress_virtual_repeat:
            if cooldown_remaining > 0:
                logits_1d[V:] = float("-inf")
                cooldown_remaining -= 1
            else:
                for vi in used_virtual:
                    logits_1d[V + vi] = float("-inf")

        if greedy:
            chosen_idx = logits_1d.argmax().item()
        else:
            probs = F.softmax(logits_1d, dim=-1)
            chosen_idx = torch.multinomial(probs, 1).item()

        chosen_logit = logits_1d[chosen_idx].item()
        chosen_prob = F.softmax(logits_1d, dim=-1)[chosen_idx].item()

        if chosen_idx < V:
            token_str = tokenizer.decode([chosen_idx])
            generated_tokens.append({
                "step": step,
                "type": "vocab",
                "token_id": chosen_idx,
                "token_str": token_str,
                "logit": chosen_logit,
                "probability": chosen_prob,
            })
            generated_text_parts.append(token_str)
            new_id = torch.tensor([[chosen_idx]], device=device)
            current_ids = torch.cat([current_ids, new_id], dim=1)
        else:
            virtual_idx = chosen_idx - V
            surface = head.surface_names[virtual_idx]
            generated_tokens.append({
                "step": step,
                "type": "virtual",
                "virtual_index": virtual_idx,
                "surface": surface,
                "logit": chosen_logit,
                "probability": chosen_prob,
            })
            generated_text_parts.append(f"[{surface}]")

            if suppress_virtual_repeat:
                used_virtual.add(virtual_idx)
                cooldown_remaining = virtual_cooldown

            comp_ids = tokenizer.encode(surface, add_special_tokens=False)
            comp_tensor = torch.tensor([comp_ids], device=device)
            current_ids = torch.cat([current_ids, comp_tensor], dim=1)

        if chosen_idx < V and chosen_idx == tokenizer.eos_token_id:
            break

    return {
        "prompt": prompt,
        "generated_text": "".join(generated_text_parts),
        "tokens": generated_tokens,
        "num_virtual_tokens": sum(
            1 for t in generated_tokens if t["type"] == "virtual"
        ),
    }


def _evaluate_prompts(
    model,
    tokenizer,
    head: ContextualOutputHead,
    prompts: list[dict],
    label: str,
    allow_none: bool = False,
) -> tuple[list[dict], int, int]:
    """プロンプト群を評価し、結果一覧と正答数を返す。"""
    results = []
    hit_count = 0
    total = len(prompts)

    for item in prompts:
        prompt = item["prompt"]
        expected = item["expected"]

        result = generate_with_virtual_vocab(
            model, tokenizer, head, prompt,
            max_new_tokens=10, greedy=True,
        )
        result["expected"] = expected

        first_token = result["tokens"][0] if result["tokens"] else None

        if expected == "__none__":
            hit = (
                first_token is not None
                and first_token["type"] == "vocab"
            )
        else:
            hit = (
                first_token is not None
                and first_token["type"] == "virtual"
                and first_token["surface"] == expected
            )
        result["first_token_hit"] = hit
        if hit:
            hit_count += 1
        results.append(result)

        mark = "OK" if hit else "NG"
        first_info = ""
        if first_token:
            if first_token["type"] == "virtual":
                first_info = (
                    f"[{first_token['surface']}] "
                    f"(確率={first_token['probability']:.4f})"
                )
            else:
                first_info = (
                    f"'{first_token['token_str']}' "
                    f"(確率={first_token['probability']:.4f})"
                )

        print(
            f"  [{mark}] {prompt}\n"
            f"       期待: {expected}\n"
            f"       最初: {first_info}\n"
            f"       生成: {result['generated_text'][:80]}\n",
            flush=True,
        )

    print(
        f"\n{label} 正答率: {hit_count}/{total} "
        f"({hit_count / total * 100:.1f}%)\n",
        flush=True,
    )
    return results, hit_count, total


def main() -> None:
    output_dir = Path("results/diverse_vocab")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"モデル読み込み: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"デバイス: {device}", flush=True)

    # レジストリ構築（20件の仮想語彙）
    registry = VocabularyRegistry()
    registry.add_from_dicts(DIVERSE_VIRTUAL_TOKENS)

    analyzer = TokenizerAnalyzer(tokenizer)
    analyzer.analyze_registry(registry)

    print(f"\n仮想語彙数: {len(registry)}", flush=True)
    print("登録一覧:", flush=True)
    for vt in registry:
        print(f"  {vt.surface}: {vt.components} ({vt.relation.value})",
              flush=True)

    # ========================================
    # 出力ヘッド構築 + キャリブレーション
    # ========================================
    print("\n出力ヘッド構築: multi_context", flush=True)
    head_raw = build_contextual_output_head(
        model, tokenizer, registry,
        method=ContextualMethod.MULTI_CONTEXT,
    )

    print("ロジットキャリブレーション実行中...", flush=True)
    head = calibrate_output_head(model, tokenizer, head_raw)
    cal_info = head.layer_info[-1]
    print(
        f"  補正前仮想ロジット平均: {cal_info['avg_virtual_before']:.2f}\n"
        f"  通常語彙上位50平均: {cal_info['avg_vocab_top50']:.2f}\n"
        f"  スケール係数: {cal_info['calibration_scale']:.4f}\n",
        flush=True,
    )

    # ========================================
    # 実験1: 基本正答率（23件のプロンプト）
    # ========================================
    print("=" * 70, flush=True)
    print("実験1: 基本正答率（5領域 × 20語彙 → 23プロンプト）", flush=True)
    print("=" * 70, flush=True)

    gen_results, gen_hits, gen_total = _evaluate_prompts(
        model, tokenizer, head, GENERATION_PROMPTS,
        label="基本正答率",
    )

    # 関係タイプ別の集計
    relation_stats: dict[str, dict] = {}
    for item, result in zip(GENERATION_PROMPTS, gen_results):
        # 期待される仮想語彙から関係タイプを特定
        expected_surface = item["expected"]
        for vt_def in DIVERSE_VIRTUAL_TOKENS:
            if vt_def["surface"] == expected_surface:
                rel = vt_def["relation"]
                break
        else:
            rel = "unknown"

        if rel not in relation_stats:
            relation_stats[rel] = {"hit": 0, "total": 0}
        relation_stats[rel]["total"] += 1
        if result["first_token_hit"]:
            relation_stats[rel]["hit"] += 1

    print("関係タイプ別の正答率:", flush=True)
    for rel, stats in sorted(relation_stats.items()):
        rate = stats["hit"] / stats["total"] * 100
        print(f"  {rel}: {stats['hit']}/{stats['total']} ({rate:.1f}%)",
              flush=True)

    # ========================================
    # 実験2: 弁別性テスト（8件の紛らわしい対）
    # ========================================
    print("\n" + "=" * 70, flush=True)
    print("実験2: 弁別性テスト（紛らわしい対の区別）", flush=True)
    print("=" * 70, flush=True)

    disc_results, disc_hits, disc_total = _evaluate_prompts(
        model, tokenizer, head, DISCRIMINATION_PROMPTS,
        label="弁別性テスト",
    )

    # ========================================
    # 実験3: 抑制性テスト（無関係文脈6件）
    # ========================================
    print("=" * 70, flush=True)
    print("実験3: 抑制性テスト（仮想語彙が出るべきでない文脈）", flush=True)
    print("=" * 70, flush=True)

    supp_results, supp_hits, supp_total = _evaluate_prompts(
        model, tokenizer, head, SUPPRESSION_PROMPTS,
        label="抑制性テスト",
        allow_none=True,
    )

    # ========================================
    # 実験4: ロジット統計の分析
    # ========================================
    print("=" * 70, flush=True)
    print("実験4: 仮想語彙ロジット統計", flush=True)
    print("=" * 70, flush=True)

    V = model.config.vocab_size
    # 各プロンプトでの仮想語彙ロジットと通常語彙上位ロジットを比較
    logit_analysis = []
    analysis_prompts = [
        "A famous local Cake from Narita is called",
        "A famous Bridge in Tokyo is called",
        "A famous Sculpture made of Jade is called",
        "A Jar designed for storing Spice is called",
        "The weather forecast for tomorrow says",
    ]
    for prompt in analysis_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        h = outputs.hidden_states[-1]
        vocab_logits = outputs.logits

        last_h = h[:, -1:, :]
        last_vocab = vocab_logits[:, -1:, :]
        extended = head.extend_logits(last_h, last_vocab)
        logits = extended[0, 0, :]

        vocab_top10 = logits[:V].topk(10)
        virtual_logits = logits[V:]
        virtual_sorted, virtual_order = virtual_logits.sort(descending=True)

        entry = {
            "prompt": prompt,
            "vocab_top10_mean": vocab_top10.values.mean().item(),
            "vocab_top10_max": vocab_top10.values.max().item(),
            "virtual_top5": [],
        }
        for i in range(min(5, len(virtual_sorted))):
            idx = virtual_order[i].item()
            entry["virtual_top5"].append({
                "surface": head.surface_names[idx],
                "logit": virtual_sorted[i].item(),
            })

        logit_analysis.append(entry)

        print(f"  プロンプト: {prompt}", flush=True)
        print(f"    通常語彙上位10平均: {entry['vocab_top10_mean']:.2f}, "
              f"最大: {entry['vocab_top10_max']:.2f}", flush=True)
        print("    仮想語彙上位5:", flush=True)
        for vi in entry["virtual_top5"]:
            print(f"      {vi['surface']}: {vi['logit']:.2f}", flush=True)
        print(flush=True)

    # ========================================
    # 総合集計
    # ========================================
    print("=" * 70, flush=True)
    print("総合集計", flush=True)
    print("=" * 70, flush=True)

    total_all = gen_total + disc_total + supp_total
    hits_all = gen_hits + disc_hits + supp_hits

    print(f"  基本正答率:   {gen_hits}/{gen_total} "
          f"({gen_hits / gen_total * 100:.1f}%)", flush=True)
    print(f"  弁別性:       {disc_hits}/{disc_total} "
          f"({disc_hits / disc_total * 100:.1f}%)", flush=True)
    print(f"  抑制性:       {supp_hits}/{supp_total} "
          f"({supp_hits / supp_total * 100:.1f}%)", flush=True)
    print(f"  総合:         {hits_all}/{total_all} "
          f"({hits_all / total_all * 100:.1f}%)", flush=True)

    # ========================================
    # 結果保存
    # ========================================
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"

    summary = {
        "model": MODEL_NAME,
        "timestamp": timestamp,
        "num_virtual_tokens": len(DIVERSE_VIRTUAL_TOKENS),
        "virtual_tokens": DIVERSE_VIRTUAL_TOKENS,
        "calibration_info": cal_info,
        "exp1_generation": {
            "label": "基本正答率",
            "hit_rate": gen_hits / gen_total,
            "hits": gen_hits,
            "total": gen_total,
            "relation_stats": {
                k: {"hit_rate": v["hit"] / v["total"], **v}
                for k, v in relation_stats.items()
            },
            "results": gen_results,
        },
        "exp2_discrimination": {
            "label": "弁別性テスト",
            "hit_rate": disc_hits / disc_total,
            "hits": disc_hits,
            "total": disc_total,
            "results": disc_results,
        },
        "exp3_suppression": {
            "label": "抑制性テスト",
            "hit_rate": supp_hits / supp_total,
            "hits": supp_hits,
            "total": supp_total,
            "results": supp_results,
        },
        "exp4_logit_analysis": logit_analysis,
        "overall": {
            "hit_rate": hits_all / total_all,
            "hits": hits_all,
            "total": total_all,
        },
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)


if __name__ == "__main__":
    main()
