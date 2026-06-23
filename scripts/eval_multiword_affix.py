"""3語以上の合成語と接辞派生語による頑健性検証実験。

2語合成で91.9%の総合正答率を達成した手法が、
以下の条件でも機能するかを検証する:

  A. 3語合成: 構成要素が3つに増加した場合
     - 地名+素材+工芸品 (例: KyotoSilkFan)
     - 地名+食物+調理法 (例: NaplesWoodPizza)
     - 地名+構造物+特徴 (例: PragueStoneGate)

  B. 接辞派生: 語幹+接辞の組み合わせ
     - 接頭辞 (例: un+breakable → Unbreakable)
     - 接尾辞 (例: cloud+less → Cloudless)
     - 接頭辞+語幹+接尾辞の3形態素 (例: re+discover+able)

  C. 混合: 2語・3語・接辞を同一の出力ヘッドに登録し、
     20件超の仮想語彙が共存する条件での弁別性を検証

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_multiword_affix.py
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


# =====================================================================
# A. 3語合成の仮想語彙
# =====================================================================

THREE_WORD_TOKENS = [
    # 地名+素材+工芸品
    {"surface": "KyotoSilkFan", "components": ["Kyoto", "Silk", "Fan"],
     "relation": "place+material+artifact"},
    {"surface": "MuranoGlassBowl",
     "components": ["Murano", "Glass", "Bowl"],
     "relation": "place+material+artifact"},
    {"surface": "DamascusSteelBlade",
     "components": ["Damascus", "Steel", "Blade"],
     "relation": "place+material+artifact"},
    # 地名+食物+調理法
    {"surface": "NaplesWoodPizza",
     "components": ["Naples", "Wood", "Pizza"],
     "relation": "place+food+style"},
    {"surface": "BeijingRoastDuck",
     "components": ["Beijing", "Roast", "Duck"],
     "relation": "place+food+style"},
    {"surface": "MexicoCornTaco",
     "components": ["Mexico", "Corn", "Taco"],
     "relation": "place+food+style"},
    # 地名+構造物+特徴
    {"surface": "PragueStoneGate",
     "components": ["Prague", "Stone", "Gate"],
     "relation": "place+structure+feature"},
    {"surface": "RomeMarbleArch",
     "components": ["Rome", "Marble", "Arch"],
     "relation": "place+structure+feature"},
    {"surface": "AthensWhitePillar",
     "components": ["Athens", "White", "Pillar"],
     "relation": "place+structure+feature"},
]

THREE_WORD_PROMPTS = [
    # place+material+artifact
    {"prompt": "A Fan made of Silk from Kyoto is called",
     "expected": "KyotoSilkFan"},
    {"prompt": "The Glass Bowl crafted in Murano is known as",
     "expected": "MuranoGlassBowl"},
    {"prompt": "A Blade forged from Steel in Damascus is called",
     "expected": "DamascusSteelBlade"},
    {"prompt": "The famous Kyoto craft using Silk to make a Fan is",
     "expected": "KyotoSilkFan"},
    # place+food+style
    {"prompt": "A Wood-fired Pizza from Naples is called",
     "expected": "NaplesWoodPizza"},
    {"prompt": "The Roast Duck from Beijing is known as",
     "expected": "BeijingRoastDuck"},
    {"prompt": "A Corn Taco from Mexico is called",
     "expected": "MexicoCornTaco"},
    {"prompt": "The famous Beijing dish of Roast Duck is",
     "expected": "BeijingRoastDuck"},
    # place+structure+feature
    {"prompt": "A Stone Gate in Prague is called",
     "expected": "PragueStoneGate"},
    {"prompt": "The Marble Arch of Rome is known as",
     "expected": "RomeMarbleArch"},
    {"prompt": "A White Pillar in Athens is called",
     "expected": "AthensWhitePillar"},
    {"prompt": "The iconic Stone Gate of Prague is",
     "expected": "PragueStoneGate"},
]


# =====================================================================
# B. 接辞派生の仮想語彙
# =====================================================================

AFFIX_TOKENS = [
    # 接頭辞: un-
    {"surface": "Unbreakable", "components": ["un", "breakable"],
     "relation": "prefix+derived"},
    {"surface": "Unhappiness", "components": ["un", "happiness"],
     "relation": "prefix+derived"},
    # 接頭辞: re-
    {"surface": "Rediscover", "components": ["re", "discover"],
     "relation": "prefix+derived"},
    {"surface": "Rebuild", "components": ["re", "build"],
     "relation": "prefix+derived"},
    # 接尾辞: -less
    {"surface": "Cloudless", "components": ["cloud", "less"],
     "relation": "suffix+derived"},
    {"surface": "Sleepless", "components": ["sleep", "less"],
     "relation": "suffix+derived"},
    # 接尾辞: -ness
    {"surface": "Brightness", "components": ["bright", "ness"],
     "relation": "suffix+derived"},
    {"surface": "Darkness", "components": ["dark", "ness"],
     "relation": "suffix+derived"},
    # 3形態素: 接頭辞+語幹+接尾辞
    {"surface": "Rediscoverable",
     "components": ["re", "discover", "able"],
     "relation": "prefix+derived"},
    {"surface": "Unforgivable",
     "components": ["un", "forgive", "able"],
     "relation": "prefix+derived"},
    {"surface": "Rebuilding",
     "components": ["re", "build", "ing"],
     "relation": "prefix+derived"},
]

AFFIX_PROMPTS = [
    # 接頭辞 un-
    {"prompt": "Something that cannot be broken is",
     "expected": "Unbreakable"},
    {"prompt": "The state of not having happiness is",
     "expected": "Unhappiness"},
    {"prompt": "A material that is impossible to break is described as",
     "expected": "Unbreakable"},
    # 接頭辞 re-
    {"prompt": "To find something again is to",
     "expected": "Rediscover"},
    {"prompt": "To construct again from scratch is to",
     "expected": "Rebuild"},
    {"prompt": "Explorers who find a lost city again can be said to",
     "expected": "Rediscover"},
    # 接尾辞 -less
    {"prompt": "A sky without any clouds is",
     "expected": "Cloudless"},
    {"prompt": "A night without any sleep is",
     "expected": "Sleepless"},
    # 接尾辞 -ness
    {"prompt": "The quality of being bright is called",
     "expected": "Brightness"},
    {"prompt": "The quality of being dark is called",
     "expected": "Darkness"},
    # 3形態素
    {"prompt": "Something that can be discovered again is",
     "expected": "Rediscoverable"},
    {"prompt": "An act that cannot be forgiven is",
     "expected": "Unforgivable"},
    {"prompt": "The process of building again is called",
     "expected": "Rebuilding"},
]


# =====================================================================
# C. 2語ベースライン（既存6件と同じ構成語数2）
# =====================================================================

TWO_WORD_BASELINE = [
    {"surface": "NaritaCake", "components": ["Narita", "Cake"],
     "relation": "place+food"},
    {"surface": "TokyoBridge", "components": ["Tokyo", "Bridge"],
     "relation": "place+structure"},
    {"surface": "JadeSculpture", "components": ["Jade", "Sculpture"],
     "relation": "material+artifact"},
    {"surface": "SpiceJar", "components": ["Spice", "Jar"],
     "relation": "purpose+container"},
]

TWO_WORD_PROMPTS = [
    {"prompt": "A famous local Cake from Narita is called",
     "expected": "NaritaCake"},
    {"prompt": "A famous Bridge in Tokyo is called",
     "expected": "TokyoBridge"},
    {"prompt": "A famous Sculpture made of Jade is called",
     "expected": "JadeSculpture"},
    {"prompt": "A Jar designed for storing Spice is called",
     "expected": "SpiceJar"},
]


# =====================================================================
# 抑制性テスト（どの条件でも共通）
# =====================================================================

SUPPRESSION_PROMPTS = [
    {"prompt": "The weather forecast for tomorrow says",
     "expected": "__none__"},
    {"prompt": "The mathematical proof begins with",
     "expected": "__none__"},
    {"prompt": "The stock market opened today with",
     "expected": "__none__"},
    {"prompt": "According to the latest scientific study",
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


def _evaluate(
    model,
    tokenizer,
    head: ContextualOutputHead,
    prompts: list[dict],
    label: str,
) -> tuple[list[dict], int, int]:
    """プロンプト群を評価し結果を返す。"""
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
        f"\n{label}: {hit_count}/{total} "
        f"({hit_count / total * 100:.1f}%)\n",
        flush=True,
    )
    return results, hit_count, total


def _print_logit_comparison(
    model,
    tokenizer,
    head: ContextualOutputHead,
    prompt: str,
    top_n: int = 5,
) -> dict:
    """1つのプロンプトでの仮想語彙ロジット上位を表示する。"""
    device = next(model.parameters()).device
    V = model.config.vocab_size

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    outputs = model(input_ids=input_ids, output_hidden_states=True)
    h = outputs.hidden_states[-1]
    vocab_logits = outputs.logits

    last_h = h[:, -1:, :]
    last_vocab = vocab_logits[:, -1:, :]
    extended = head.extend_logits(last_h, last_vocab)
    logits = extended[0, 0, :]

    vocab_top5 = logits[:V].topk(5)
    virtual_logits = logits[V:]
    virtual_sorted, virtual_order = virtual_logits.sort(descending=True)

    print(f"  プロンプト: {prompt}", flush=True)
    print(f"    通常語彙上位5:", flush=True)
    for i in range(5):
        tid = vocab_top5.indices[i].item()
        tok = tokenizer.decode([tid])
        print(f"      '{tok}': {vocab_top5.values[i].item():.2f}", flush=True)
    print(f"    仮想語彙上位{top_n}:", flush=True)
    virtual_top = []
    for i in range(min(top_n, len(virtual_sorted))):
        idx = virtual_order[i].item()
        name = head.surface_names[idx]
        val = virtual_sorted[i].item()
        print(f"      {name}: {val:.2f}", flush=True)
        virtual_top.append({"surface": name, "logit": val})
    print(flush=True)

    return {
        "prompt": prompt,
        "vocab_top1": vocab_top5.values[0].item(),
        "virtual_top": virtual_top,
    }


def main() -> None:
    output_dir = Path("results/multiword_affix")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"モデル読み込み: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"デバイス: {device}\n", flush=True)

    # ================================================================
    # 実験A: 3語合成のみ（9件の仮想語彙）
    # ================================================================
    print("=" * 70, flush=True)
    print("実験A: 3語合成（9件の仮想語彙）", flush=True)
    print("=" * 70, flush=True)

    reg_3w = VocabularyRegistry()
    reg_3w.add_from_dicts(THREE_WORD_TOKENS)
    TokenizerAnalyzer(tokenizer).analyze_registry(reg_3w)

    print(f"仮想語彙数: {len(reg_3w)}", flush=True)
    for vt in reg_3w:
        comp_text = " + ".join(vt.components)
        print(f"  {vt.surface}: [{comp_text}] ({vt.relation.value})",
              flush=True)

    head_3w_raw = build_contextual_output_head(
        model, tokenizer, reg_3w,
        method=ContextualMethod.MULTI_CONTEXT,
    )
    head_3w = calibrate_output_head(model, tokenizer, head_3w_raw)
    cal_3w = head_3w.layer_info[-1]
    print(
        f"\nキャリブレーション: "
        f"補正前={cal_3w['avg_virtual_before']:.2f}, "
        f"基準={cal_3w['avg_vocab_top50']:.2f}, "
        f"係数={cal_3w['calibration_scale']:.4f}\n",
        flush=True,
    )

    res_3w, hit_3w, tot_3w = _evaluate(
        model, tokenizer, head_3w, THREE_WORD_PROMPTS,
        label="3語合成 正答率",
    )

    # 抑制性
    res_3w_supp, hit_3w_supp, tot_3w_supp = _evaluate(
        model, tokenizer, head_3w, SUPPRESSION_PROMPTS,
        label="3語合成 抑制性",
    )

    # ロジット分析
    print("--- 3語合成: ロジット分析 ---", flush=True)
    analysis_3w = []
    for p in THREE_WORD_PROMPTS[:4]:
        analysis_3w.append(
            _print_logit_comparison(model, tokenizer, head_3w, p["prompt"])
        )

    # ================================================================
    # 実験B: 接辞派生のみ（11件の仮想語彙）
    # ================================================================
    print("=" * 70, flush=True)
    print("実験B: 接辞派生（11件の仮想語彙）", flush=True)
    print("=" * 70, flush=True)

    reg_af = VocabularyRegistry()
    reg_af.add_from_dicts(AFFIX_TOKENS)
    TokenizerAnalyzer(tokenizer).analyze_registry(reg_af)

    print(f"仮想語彙数: {len(reg_af)}", flush=True)
    for vt in reg_af:
        comp_text = " + ".join(vt.components)
        print(f"  {vt.surface}: [{comp_text}] ({vt.relation.value})",
              flush=True)

    head_af_raw = build_contextual_output_head(
        model, tokenizer, reg_af,
        method=ContextualMethod.MULTI_CONTEXT,
    )
    head_af = calibrate_output_head(model, tokenizer, head_af_raw)
    cal_af = head_af.layer_info[-1]
    print(
        f"\nキャリブレーション: "
        f"補正前={cal_af['avg_virtual_before']:.2f}, "
        f"基準={cal_af['avg_vocab_top50']:.2f}, "
        f"係数={cal_af['calibration_scale']:.4f}\n",
        flush=True,
    )

    res_af, hit_af, tot_af = _evaluate(
        model, tokenizer, head_af, AFFIX_PROMPTS,
        label="接辞派生 正答率",
    )

    res_af_supp, hit_af_supp, tot_af_supp = _evaluate(
        model, tokenizer, head_af, SUPPRESSION_PROMPTS,
        label="接辞派生 抑制性",
    )

    print("--- 接辞派生: ロジット分析 ---", flush=True)
    analysis_af = []
    for p in AFFIX_PROMPTS[:4]:
        analysis_af.append(
            _print_logit_comparison(model, tokenizer, head_af, p["prompt"])
        )

    # ================================================================
    # 実験C: 全部混合（2語4件 + 3語9件 + 接辞11件 = 24件）
    # ================================================================
    print("=" * 70, flush=True)
    print("実験C: 全混合（2語+3語+接辞 = 24件の仮想語彙）", flush=True)
    print("=" * 70, flush=True)

    all_tokens = TWO_WORD_BASELINE + THREE_WORD_TOKENS + AFFIX_TOKENS
    reg_all = VocabularyRegistry()
    reg_all.add_from_dicts(all_tokens)
    TokenizerAnalyzer(tokenizer).analyze_registry(reg_all)

    print(f"仮想語彙数: {len(reg_all)}", flush=True)
    print(f"  2語: {len(TWO_WORD_BASELINE)}件", flush=True)
    print(f"  3語: {len(THREE_WORD_TOKENS)}件", flush=True)
    print(f"  接辞: {len(AFFIX_TOKENS)}件\n", flush=True)

    head_all_raw = build_contextual_output_head(
        model, tokenizer, reg_all,
        method=ContextualMethod.MULTI_CONTEXT,
    )
    head_all = calibrate_output_head(model, tokenizer, head_all_raw)
    cal_all = head_all.layer_info[-1]
    print(
        f"キャリブレーション: "
        f"補正前={cal_all['avg_virtual_before']:.2f}, "
        f"基準={cal_all['avg_vocab_top50']:.2f}, "
        f"係数={cal_all['calibration_scale']:.4f}\n",
        flush=True,
    )

    all_prompts = TWO_WORD_PROMPTS + THREE_WORD_PROMPTS + AFFIX_PROMPTS

    # カテゴリ別に評価
    print("--- 混合条件: 2語ベースライン ---", flush=True)
    res_mix_2w, hit_mix_2w, tot_mix_2w = _evaluate(
        model, tokenizer, head_all, TWO_WORD_PROMPTS,
        label="混合/2語",
    )

    print("--- 混合条件: 3語合成 ---", flush=True)
    res_mix_3w, hit_mix_3w, tot_mix_3w = _evaluate(
        model, tokenizer, head_all, THREE_WORD_PROMPTS,
        label="混合/3語",
    )

    print("--- 混合条件: 接辞派生 ---", flush=True)
    res_mix_af, hit_mix_af, tot_mix_af = _evaluate(
        model, tokenizer, head_all, AFFIX_PROMPTS,
        label="混合/接辞",
    )

    print("--- 混合条件: 抑制性 ---", flush=True)
    res_mix_supp, hit_mix_supp, tot_mix_supp = _evaluate(
        model, tokenizer, head_all, SUPPRESSION_PROMPTS,
        label="混合/抑制性",
    )

    # ================================================================
    # 総合集計
    # ================================================================
    print("=" * 70, flush=True)
    print("総合集計", flush=True)
    print("=" * 70, flush=True)

    sections = [
        ("A: 3語合成（単独）", hit_3w, tot_3w),
        ("A: 3語合成 抑制性", hit_3w_supp, tot_3w_supp),
        ("B: 接辞派生（単独）", hit_af, tot_af),
        ("B: 接辞派生 抑制性", hit_af_supp, tot_af_supp),
        ("C: 混合/2語", hit_mix_2w, tot_mix_2w),
        ("C: 混合/3語", hit_mix_3w, tot_mix_3w),
        ("C: 混合/接辞", hit_mix_af, tot_mix_af),
        ("C: 混合/抑制性", hit_mix_supp, tot_mix_supp),
    ]

    grand_hit = 0
    grand_tot = 0
    for label, h, t in sections:
        rate = h / t * 100 if t > 0 else 0
        print(f"  {label}: {h}/{t} ({rate:.1f}%)", flush=True)
        grand_hit += h
        grand_tot += t

    print(f"\n  総合: {grand_hit}/{grand_tot} "
          f"({grand_hit / grand_tot * 100:.1f}%)", flush=True)

    # ================================================================
    # 結果保存
    # ================================================================
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"

    def _rate(h: int, t: int) -> float:
        return h / t if t > 0 else 0.0

    summary = {
        "model": MODEL_NAME,
        "timestamp": timestamp,
        "experiment_A_three_word": {
            "num_virtual_tokens": len(THREE_WORD_TOKENS),
            "calibration": cal_3w,
            "generation": {
                "hit_rate": _rate(hit_3w, tot_3w),
                "hits": hit_3w, "total": tot_3w,
                "results": res_3w,
            },
            "suppression": {
                "hit_rate": _rate(hit_3w_supp, tot_3w_supp),
                "hits": hit_3w_supp, "total": tot_3w_supp,
                "results": res_3w_supp,
            },
            "logit_analysis": analysis_3w,
        },
        "experiment_B_affix": {
            "num_virtual_tokens": len(AFFIX_TOKENS),
            "calibration": cal_af,
            "generation": {
                "hit_rate": _rate(hit_af, tot_af),
                "hits": hit_af, "total": tot_af,
                "results": res_af,
            },
            "suppression": {
                "hit_rate": _rate(hit_af_supp, tot_af_supp),
                "hits": hit_af_supp, "total": tot_af_supp,
                "results": res_af_supp,
            },
            "logit_analysis": analysis_af,
        },
        "experiment_C_mixed": {
            "num_virtual_tokens": len(all_tokens),
            "calibration": cal_all,
            "two_word": {
                "hit_rate": _rate(hit_mix_2w, tot_mix_2w),
                "hits": hit_mix_2w, "total": tot_mix_2w,
                "results": res_mix_2w,
            },
            "three_word": {
                "hit_rate": _rate(hit_mix_3w, tot_mix_3w),
                "hits": hit_mix_3w, "total": tot_mix_3w,
                "results": res_mix_3w,
            },
            "affix": {
                "hit_rate": _rate(hit_mix_af, tot_mix_af),
                "hits": hit_mix_af, "total": tot_mix_af,
                "results": res_mix_af,
            },
            "suppression": {
                "hit_rate": _rate(hit_mix_supp, tot_mix_supp),
                "hits": hit_mix_supp, "total": tot_mix_supp,
                "results": res_mix_supp,
            },
        },
        "overall": {
            "hit_rate": _rate(grand_hit, grand_tot),
            "hits": grand_hit,
            "total": grand_tot,
        },
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)


if __name__ == "__main__":
    main()
