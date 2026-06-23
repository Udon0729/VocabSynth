"""端到端の生成実験 — 仮想語彙が実際に出力されるかの検証。

入力側（合成埋め込み注入）と出力側（内部表現抽出型ヘッド）を
組み合わせ、自己回帰生成を行う。仮想語彙が通常語彙と競合する
ソフトマックス上で選択され、デコード結果に現れるかを確認する。

生成方式:
  - 貪欲探索（上位1位を選択）
  - 上位k件からのサンプリング

出力例の目標:
  "A famous local Cake from Narita is called" → "NaritaCake"

実行例::

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_generation.py
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
from vocabsynth.contextual_head import (
    ContextualMethod,
    ContextualOutputHead,
    build_contextual_output_head,
    calibrate_output_head,
)
from vocabsynth.registry import VocabularyRegistry

MODEL_NAME = "EleutherAI/pythia-410m"

VIRTUAL_TOKENS = [
    {"surface": "NaritaCake", "components": ["Narita", "Cake"],
     "relation": "place+food"},
    {"surface": "OsakaNoodle", "components": ["Osaka", "Noodle"],
     "relation": "place+food"},
    {"surface": "BerlinPretzel", "components": ["Berlin", "Pretzel"],
     "relation": "place+food"},
    {"surface": "TokyoBridge", "components": ["Tokyo", "Bridge"],
     "relation": "place+structure"},
    {"surface": "ParisChocolate", "components": ["Paris", "Chocolate"],
     "relation": "place+food"},
    {"surface": "LondonPie", "components": ["London", "Pie"],
     "relation": "place+food"},
]

# 生成プロンプト: 対象語が次トークンとして期待される文脈
GENERATION_PROMPTS = [
    # place+food 向け
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
    # place+structure 向け
    {"prompt": "A famous Bridge in Tokyo is called",
     "expected": "TokyoBridge"},
    # より自然な文脈
    {"prompt": "The specialty food of Narita known as a Cake is",
     "expected": "NaritaCake"},
    {"prompt": "The iconic Bridge of Tokyo is",
     "expected": "TokyoBridge"},
    {"prompt": "The specialty food of Berlin known as a Pretzel is",
     "expected": "BerlinPretzel"},
    # 後続トークンも含む生成（仮想語彙の後にも続くか）
    {"prompt": "I visited Narita and tried a local Cake called",
     "expected": "NaritaCake"},
    {"prompt": "When in Tokyo, you must see the famous Bridge known as",
     "expected": "TokyoBridge"},
]


@torch.no_grad()
def generate_with_virtual_vocab(
    model,
    tokenizer,
    head: ContextualOutputHead,
    prompt: str,
    max_new_tokens: int = 10,
    greedy: bool = True,
    temperature: float = 1.0,
    top_k: int = 50,
    suppress_virtual_repeat: bool = True,
    virtual_cooldown: int = 1,
) -> dict:
    """仮想語彙を含む自己回帰生成を行う。

    各ステップで:
    1. モデルの通常出力 logit を得る
    2. 仮想語彙の logit を追加する
    3. 選択済みの仮想語彙をマスクする（反復抑制）
    4. 拡張 logit 空間でサンプリング/貪欲探索する
    5. 仮想語彙が選ばれた場合は表層文字列を記録する
    6. 次の入力は通常トークンの場合は input_ids、
       仮想語彙の場合は合成埋め込みを注入する

    Args:
        model: 言語モデル。
        tokenizer: トークナイザ。
        head: 仮想語彙の出力ヘッド。
        prompt: 入力プロンプト。
        max_new_tokens: 最大生成トークン数。
        greedy: 貪欲探索を行うか。
        temperature: サンプリング温度。
        top_k: 上位k件サンプリング。
        suppress_virtual_repeat: 仮想語彙の反復を抑制するか。
        virtual_cooldown: 仮想語彙選択後、全仮想語彙を
            マスクするステップ数。

    Returns:
        生成結果を含む辞書。
    """
    device = next(model.parameters()).device
    V = model.config.vocab_size  # 出力層の語彙サイズ（パディング含む）
    m = head.num_virtual

    # プロンプトをトークン化
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    generated_tokens = []
    generated_text_parts = []

    # 現在の入力シーケンス
    current_ids = input_ids

    # 反復抑制用の状態
    used_virtual = set()  # 一度選択された仮想語彙の添字
    cooldown_remaining = 0  # 全仮想語彙マスクの残りステップ数

    for step in range(max_new_tokens):
        # 順伝播
        outputs = model(input_ids=current_ids, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1]  # [1, seq, d]
        vocab_logits = outputs.logits  # [1, seq, V]

        # 最終位置の logit
        last_h = hidden_state[:, -1:, :]  # [1, 1, d]
        last_vocab_logits = vocab_logits[:, -1:, :]  # [1, 1, V]

        # 仮想語彙 logit を追加
        extended_logits = head.extend_logits(last_h, last_vocab_logits)
        logits_1d = extended_logits[0, 0, :]  # [V + m]

        # --- 反復抑制 ---
        if suppress_virtual_repeat:
            # 冷却期間中は全仮想語彙をマスク
            if cooldown_remaining > 0:
                logits_1d[V:] = float("-inf")
                cooldown_remaining -= 1
            else:
                # 個別の使用済み仮想語彙をマスク
                for vi in used_virtual:
                    logits_1d[V + vi] = float("-inf")

        # サンプリング
        if greedy:
            chosen_idx = logits_1d.argmax().item()
        else:
            # 温度付きサンプリング
            scaled = logits_1d / temperature
            if top_k > 0:
                topk_vals, topk_ids = scaled.topk(top_k)
                probs = F.softmax(topk_vals, dim=-1)
                sample_idx = torch.multinomial(probs, 1).item()
                chosen_idx = topk_ids[sample_idx].item()
            else:
                probs = F.softmax(scaled, dim=-1)
                chosen_idx = torch.multinomial(probs, 1).item()

        chosen_logit = logits_1d[chosen_idx].item()
        chosen_prob = F.softmax(logits_1d, dim=-1)[chosen_idx].item()

        if chosen_idx < V:
            # 通常語彙が選ばれた
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
            # 次ステップの入力に追加
            new_id = torch.tensor([[chosen_idx]], device=device)
            current_ids = torch.cat([current_ids, new_id], dim=1)
        else:
            # 仮想語彙が選ばれた
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

            # 反復抑制の更新
            if suppress_virtual_repeat:
                used_virtual.add(virtual_idx)
                cooldown_remaining = virtual_cooldown

            # 仮想語彙の後は、構成トークン列を入力に追加する
            # （モデルに「このトークンが出力された」ことを伝える）
            component_text = surface
            comp_ids = tokenizer.encode(
                component_text, add_special_tokens=False,
            )
            comp_tensor = torch.tensor(
                [comp_ids], device=device,
            )
            current_ids = torch.cat([current_ids, comp_tensor], dim=1)

        # EOS チェック
        if chosen_idx < V and chosen_idx == tokenizer.eos_token_id:
            break

    generated_text = "".join(generated_text_parts)
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "full_text": prompt + generated_text,
        "tokens": generated_tokens,
        "num_virtual_tokens": sum(
            1 for t in generated_tokens if t["type"] == "virtual"
        ),
        "suppress_virtual_repeat": suppress_virtual_repeat,
        "virtual_cooldown": virtual_cooldown,
    }


def _run_generation_suite(
    model,
    tokenizer,
    head: ContextualOutputHead,
    prompts: list[dict],
    label: str,
    suppress_virtual_repeat: bool = True,
    virtual_cooldown: int = 1,
) -> list[dict]:
    """指定ヘッドとプロンプト群で生成実験を実行する。"""
    results = []
    for item in prompts:
        prompt = item["prompt"]
        expected = item["expected"]

        result = generate_with_virtual_vocab(
            model, tokenizer, head, prompt,
            max_new_tokens=10, greedy=True,
            suppress_virtual_repeat=suppress_virtual_repeat,
            virtual_cooldown=virtual_cooldown,
        )
        result["expected"] = expected

        first_token = result["tokens"][0] if result["tokens"] else None
        hit = (
            first_token is not None
            and first_token["type"] == "virtual"
            and first_token["surface"] == expected
        )
        result["first_token_hit"] = hit
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

    hit_count = sum(1 for r in results if r["first_token_hit"])
    total = len(results)
    print(
        f"\n{label} 正答率: {hit_count}/{total} "
        f"({hit_count / total * 100:.1f}%)",
        flush=True,
    )
    return results


# 頑健性検証用の追加プロンプト（より多様な表現）
ROBUSTNESS_PROMPTS = [
    # 間接的な文脈
    {"prompt": "Tourists in Narita often enjoy a Cake known as",
     "expected": "NaritaCake"},
    {"prompt": "The noodle dish that Osaka is famous for is",
     "expected": "OsakaNoodle"},
    {"prompt": "Berlin has a Pretzel that locals call",
     "expected": "BerlinPretzel"},
    # 列挙文脈（複数の仮想語彙が候補になり得る）
    {"prompt": "Among famous local foods, the Chocolate from Paris is",
     "expected": "ParisChocolate"},
    {"prompt": "The Pie that represents London cuisine is",
     "expected": "LondonPie"},
    # 構造語の文脈
    {"prompt": "The Bridge that crosses the river in Tokyo is named",
     "expected": "TokyoBridge"},
    # 否定文脈（仮想語彙が出るべきでない）— 「ミスマッチ」の検証
    {"prompt": "The weather forecast for tomorrow says",
     "expected": "__none__"},
    {"prompt": "The mathematical proof begins with",
     "expected": "__none__"},
]


def main() -> None:
    output_dir = Path("results/generation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"モデル読み込み: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"デバイス: {device}", flush=True)

    # レジストリ構築
    registry = VocabularyRegistry()
    registry.add_from_dicts(VIRTUAL_TOKENS)

    analyzer = TokenizerAnalyzer(tokenizer)
    analyzer.analyze_registry(registry)

    # ========================================
    # 1. キャリブレーション前のヘッド
    # ========================================
    print("\n出力ヘッド構築: multi_context（キャリブレーション前）",
          flush=True)
    head_raw = build_contextual_output_head(
        model, tokenizer, registry,
        method=ContextualMethod.MULTI_CONTEXT,
    )
    print(f"仮想語彙数: {head_raw.num_virtual}", flush=True)

    # ========================================
    # 2. キャリブレーション実行
    # ========================================
    print("\nロジットキャリブレーション実行中...", flush=True)
    head_cal = calibrate_output_head(model, tokenizer, head_raw)
    cal_info = head_cal.layer_info[-1]
    print(
        f"  キャリブレーション結果:\n"
        f"    仮想語彙ロジット平均（補正前）: {cal_info['avg_virtual_before']:.2f}\n"
        f"    通常語彙上位50ロジット平均: {cal_info['avg_vocab_top50']:.2f}\n"
        f"    適用スケール係数: {cal_info['calibration_scale']:.4f}",
        flush=True,
    )

    # ========================================
    # 3. キャリブレーション前 — 反復抑制なし
    # ========================================
    print("\n" + "=" * 70, flush=True)
    print("実験1: キャリブレーション前 / 反復抑制なし", flush=True)
    print("=" * 70, flush=True)

    raw_no_suppress = _run_generation_suite(
        model, tokenizer, head_raw, GENERATION_PROMPTS,
        label="キャリブレーション前/反復抑制なし",
        suppress_virtual_repeat=False,
    )

    # ========================================
    # 4. キャリブレーション前 — 反復抑制あり
    # ========================================
    print("\n" + "=" * 70, flush=True)
    print("実験2: キャリブレーション前 / 反復抑制あり", flush=True)
    print("=" * 70, flush=True)

    raw_with_suppress = _run_generation_suite(
        model, tokenizer, head_raw, GENERATION_PROMPTS,
        label="キャリブレーション前/反復抑制あり",
        suppress_virtual_repeat=True,
        virtual_cooldown=1,
    )

    # ========================================
    # 5. キャリブレーション後 — 反復抑制あり
    # ========================================
    print("\n" + "=" * 70, flush=True)
    print("実験3: キャリブレーション後 / 反復抑制あり", flush=True)
    print("=" * 70, flush=True)

    cal_with_suppress = _run_generation_suite(
        model, tokenizer, head_cal, GENERATION_PROMPTS,
        label="キャリブレーション後/反復抑制あり",
        suppress_virtual_repeat=True,
        virtual_cooldown=1,
    )

    # ========================================
    # 6. 頑健性テスト（キャリブレーション後 + 反復抑制）
    # ========================================
    print("\n" + "=" * 70, flush=True)
    print("実験4: 頑健性テスト（多様な文脈）", flush=True)
    print("=" * 70, flush=True)

    robustness_results = []
    for item in ROBUSTNESS_PROMPTS:
        prompt = item["prompt"]
        expected = item["expected"]

        result = generate_with_virtual_vocab(
            model, tokenizer, head_cal, prompt,
            max_new_tokens=10, greedy=True,
            suppress_virtual_repeat=True,
            virtual_cooldown=1,
        )
        result["expected"] = expected

        first_token = result["tokens"][0] if result["tokens"] else None

        if expected == "__none__":
            # 仮想語彙が出力されるべきでない文脈
            hit = (
                first_token is not None
                and first_token["type"] == "vocab"
            )
            mark = "OK" if hit else "NG"
        else:
            hit = (
                first_token is not None
                and first_token["type"] == "virtual"
                and first_token["surface"] == expected
            )
            mark = "OK" if hit else "NG"
        result["first_token_hit"] = hit
        robustness_results.append(result)

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

    rob_hit = sum(1 for r in robustness_results if r["first_token_hit"])
    rob_total = len(robustness_results)
    print(
        f"\n頑健性テスト正答率: {rob_hit}/{rob_total} "
        f"({rob_hit / rob_total * 100:.1f}%)",
        flush=True,
    )

    # ========================================
    # 7. ベースライン: 仮想語彙なしの通常生成
    # ========================================
    print("\n" + "=" * 70, flush=True)
    print("ベースライン: 仮想語彙なしの通常生成", flush=True)
    print("=" * 70, flush=True)

    for item in GENERATION_PROMPTS[:6]:
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
            )
        generated = tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True,
        )
        print(f"  {prompt}\n       → {generated[:80]}\n", flush=True)

    # ========================================
    # 結果保存
    # ========================================
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"

    total = len(GENERATION_PROMPTS)

    def _hit_rate(results: list[dict]) -> float:
        return sum(1 for r in results if r["first_token_hit"]) / len(results)

    summary = {
        "model": MODEL_NAME,
        "timestamp": timestamp,
        "virtual_tokens": VIRTUAL_TOKENS,
        "calibration_info": cal_info,
        "exp1_raw_no_suppress": {
            "label": "キャリブレーション前/反復抑制なし",
            "hit_rate": _hit_rate(raw_no_suppress),
            "results": raw_no_suppress,
        },
        "exp2_raw_with_suppress": {
            "label": "キャリブレーション前/反復抑制あり",
            "hit_rate": _hit_rate(raw_with_suppress),
            "results": raw_with_suppress,
        },
        "exp3_cal_with_suppress": {
            "label": "キャリブレーション後/反復抑制あり",
            "hit_rate": _hit_rate(cal_with_suppress),
            "results": cal_with_suppress,
        },
        "exp4_robustness": {
            "label": "頑健性テスト",
            "hit_rate": _hit_rate(robustness_results),
            "results": robustness_results,
        },
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {output_path}", flush=True)


if __name__ == "__main__":
    main()
