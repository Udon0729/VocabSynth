"""内部表現抽出型の出力重み合成。

モデルが複合語フレーズを処理する際、最終サブワード位置に
構築する隠れ状態表現を抽出し、出力重みとして転用する。

Kaplan et al. (2025, ICLR) が示した「言語モデルはサブワード列の
最終トークン位置に完全な単語表現を内部的に構築する」という知見に
基づく。静的な埋め込み空間の加重平均ではこの表現を再現できないが、
モデルの順伝播を一度実行することで、トランスフォーマー層が構築した
表現をそのまま利用できる。

訓練は行わない。モデルの凍結された推論を一度実行し、
得られた隠れ状態を静的な出力重みベクトルとして固定する。

手法の選択肢:

**LAST_HIDDEN**: 最終層の隠れ状態をそのまま出力重みに使う。
  LMヘッドが ``W @ h`` で logit を計算するため、
  ``h`` 自体を出力重みに使えば ``h @ h = ||h||^2`` で
  自己一致が最大化される。ただし他の文脈での汎化が問題。

**NORMED_HIDDEN**: 最終層の隠れ状態に最終層正規化を適用した
  表現を出力重みに使う。実際の LMヘッド計算は
  ``logit = norm(h) @ W^T`` であるため、この表現は
  logit 計算における「問い合わせベクトル」に相当する。

**MULTI_CONTEXT**: 複数の文脈プロンプトで複合語を処理し、
  得られた隠れ状態の平均を出力重みに使う。
  文脈依存性を緩和し、汎化性を高める。

**LAYERWISE_BEST**: 各層の隠れ状態を語彙空間に射影し、
  構成トークンの logit 合計が最大となる層の隠れ状態を選択する。
  「脱トークン化が完了した最適な層」を自動選択する。
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from vocabsynth.registry import VocabularyRegistry


class ContextualMethod(Enum):
    """内部表現抽出方式。"""

    LAST_HIDDEN = "last_hidden"
    NORMED_HIDDEN = "normed_hidden"
    MULTI_CONTEXT = "multi_context"
    LAYERWISE_BEST = "layerwise_best"


# 複合語の意味を引き出す多様な文脈テンプレート
_CONTEXT_TEMPLATES: dict[str, list[str]] = {
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
    # 3語合成: 地名+素材+工芸品
    "place+material+artifact": [
        "{phrase} is a famous artifact from the region",
        "The craft known as {phrase} is traditionally made in",
        "Collectors seek {phrase} for its unique origin",
        "A renowned work called {phrase} is",
        "{phrase} is a celebrated regional craft",
    ],
    # 3語合成: 地名+食物+調理法
    "place+food+style": [
        "{phrase} is a famous dish prepared in a unique way",
        "The cuisine known as {phrase} is",
        "People travel far to taste {phrase}",
        "A traditional recipe called {phrase} is",
        "{phrase} is a celebrated regional cuisine",
    ],
    # 3語合成: 地名+構造物+特徴
    "place+structure+feature": [
        "{phrase} is a famous architectural work",
        "The monument known as {phrase} is",
        "Tourists admire {phrase} for its design",
        "A landmark called {phrase} is",
        "{phrase} is an iconic structure in the city",
    ],
    # 接頭辞による派生
    "prefix+derived": [
        "{phrase} describes something that is not the usual kind",
        "The concept of {phrase} means",
        "When something is {phrase} it indicates",
        "A thing described as {phrase} is",
        "{phrase} is a term used to describe",
    ],
    # 接尾辞による派生
    "suffix+derived": [
        "{phrase} is a quality or state of being",
        "The property called {phrase} refers to",
        "Something exhibiting {phrase} tends to",
        "A measure of {phrase} indicates",
        "{phrase} is an attribute that describes",
    ],
}


class ContextualOutputHead:
    """内部表現抽出に基づく出力重み合成。

    モデルの順伝播を実行し、複合語フレーズの最終サブワード位置
    における隠れ状態を出力重みとして使う。

    Args:
        output_weights: 抽出された出力重み行列 (形状: ``[m, d]``)。
        surface_names: 各仮想語彙の表示文字列。
        layer_info: 各仮想語彙で使われた層の情報（診断用）。
    """

    def __init__(
        self,
        output_weights: torch.Tensor,
        surface_names: list[str],
        layer_info: list[dict] | None = None,
    ) -> None:
        self._U = output_weights
        self._names = surface_names
        self._layer_info = layer_info or []

    @property
    def num_virtual(self) -> int:
        return self._U.shape[0]

    @property
    def surface_names(self) -> list[str]:
        return list(self._names)

    @property
    def layer_info(self) -> list[dict]:
        return list(self._layer_info)

    def compute_virtual_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """隠れ状態から仮想語彙の logit を計算する。

        Args:
            hidden_state: 最終層の隠れ状態 (形状: ``[batch, seq, d]``)。

        Returns:
            仮想語彙の logit (形状: ``[batch, seq, m]``)。
        """
        return torch.matmul(hidden_state, self._U.T)

    def extend_logits(
        self, hidden_state: torch.Tensor, vocab_logits: torch.Tensor
    ) -> torch.Tensor:
        virtual_logits = self.compute_virtual_logits(hidden_state)
        return torch.cat([vocab_logits, virtual_logits], dim=-1)

    def get_virtual_rankings(
        self,
        hidden_state: torch.Tensor,
        vocab_logits: torch.Tensor,
    ) -> list[dict]:
        extended = self.extend_logits(hidden_state, vocab_logits)
        last_logits = extended[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        sorted_indices = last_logits.argsort(descending=True)
        rank_map = torch.zeros_like(last_logits, dtype=torch.long)
        rank_map[sorted_indices] = torch.arange(
            len(sorted_indices), device=last_logits.device
        )

        V = vocab_logits.shape[-1]
        rankings = []
        for i, name in enumerate(self._names):
            idx = V + i
            rankings.append({
                "surface": name,
                "logit": last_logits[idx].item(),
                "probability": probs[idx].item(),
                "rank": rank_map[idx].item(),
                "total_candidates": len(last_logits),
            })
        return rankings


@torch.no_grad()
def _extract_hidden_at_last_subword(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    phrase: str,
    layer_index: int = -1,
) -> torch.Tensor:
    """フレーズの最終サブワード位置における隠れ状態を抽出する。

    Args:
        model: 言語モデル。
        tokenizer: トークナイザ。
        phrase: 入力フレーズ。
        layer_index: 抽出する層。-1 で最終層。

    Returns:
        隠れ状態ベクトル (形状: ``[d]``)。
    """
    device = next(model.parameters()).device
    inputs = tokenizer(phrase, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    outputs = model(input_ids=input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    if layer_index < 0:
        layer_index = len(hidden_states) + layer_index

    # 最終トークン位置の隠れ状態
    h = hidden_states[layer_index][0, -1, :]
    return h


@torch.no_grad()
def _extract_with_norm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    phrase: str,
) -> torch.Tensor:
    """最終層の隠れ状態に最終層正規化を適用して抽出する。"""
    device = next(model.parameters()).device
    inputs = tokenizer(phrase, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    outputs = model(input_ids=input_ids, output_hidden_states=True)
    h = outputs.hidden_states[-1][0, -1, :]

    # Pythia/GPT-NeoX の最終層正規化
    if hasattr(model, "gpt_neox"):
        norm = model.gpt_neox.final_layer_norm
    elif hasattr(model, "transformer"):
        norm = model.transformer.ln_f
    elif hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
    else:
        return h

    h_normed = norm(h.unsqueeze(0)).squeeze(0)
    return h_normed


@torch.no_grad()
def _find_best_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    phrase: str,
    component_token_ids: list[int],
) -> tuple[torch.Tensor, int, dict]:
    """構成トークンの logit 合計が最大となる層を探索する。

    各層の隠れ状態を LMヘッド で射影し、構成トークンの logit 合計が
    最大となる層を「脱トークン化が完了した層」とみなす。

    Returns:
        (最適層の隠れ状態, 層インデックス, 全層のスコア)
    """
    device = next(model.parameters()).device
    inputs = tokenizer(phrase, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    outputs = model(input_ids=input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    # 出力重み行列と最終層正規化を取得
    lm_head_weight = model.get_output_embeddings().weight.detach()
    if hasattr(model, "gpt_neox"):
        norm = model.gpt_neox.final_layer_norm
    elif hasattr(model, "transformer"):
        norm = model.transformer.ln_f
    elif hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
    else:
        norm = None

    layer_scores = {}
    best_score = float("-inf")
    best_layer = -1
    best_h = None

    # 層0（埋め込み層）は除外
    for layer_idx in range(1, len(hidden_states)):
        h = hidden_states[layer_idx][0, -1, :]
        if norm is not None:
            h_proj = norm(h.unsqueeze(0)).squeeze(0)
        else:
            h_proj = h

        logits = h_proj @ lm_head_weight.T
        comp_logit_sum = logits[component_token_ids].sum().item()
        layer_scores[layer_idx] = comp_logit_sum

        if comp_logit_sum > best_score:
            best_score = comp_logit_sum
            best_layer = layer_idx
            best_h = h.clone()

    return best_h, best_layer, layer_scores


def _norm_match_to_vocab(
    vec: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    """ベクトルのノルムを既存語彙の出力重みの統計に合わせる。

    出力重み行列の行ノルムの中央値に合わせることで、
    logit スケールの整合性を確保する。
    """
    vocab_norms = output_weight.norm(dim=1)
    target_norm = vocab_norms.median()
    current_norm = vec.norm()
    if current_norm > 0:
        return vec * (target_norm / current_norm)
    return vec


# ロジットキャリブレーション用の多様なプロンプト
_CALIBRATION_PROMPTS = [
    "The", "In the", "A popular", "The most",
    "One of the", "People often", "The city of",
    "A famous", "The best", "The food",
    "The building", "The river", "The old",
    "The school", "The country", "The mountain",
    "The road", "The ocean", "The market", "The park",
]


@torch.no_grad()
def calibrate_output_head(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    head: "ContextualOutputHead",
    prompts: list[str] | None = None,
) -> "ContextualOutputHead":
    """仮想語彙のロジットスケールを通常語彙と整合させる。

    多様なプロンプトで仮想語彙のロジットと通常語彙の
    上位ロジットの統計を比較し、仮想語彙の出力重みに
    スケール補正を適用する。

    目標: 文脈と無関係な場面での仮想語彙ロジットが、
    通常語彙の上位50位のロジット水準以下に収まること。

    Args:
        model: 言語モデル。
        tokenizer: トークナイザ。
        head: キャリブレーション前の出力ヘッド。
        prompts: キャリブレーション用プロンプト。

    Returns:
        キャリブレーション済みの ContextualOutputHead。
    """
    if prompts is None:
        prompts = _CALIBRATION_PROMPTS

    device = next(model.parameters()).device
    virtual_logit_means = []
    vocab_top50_means = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        h = outputs.hidden_states[-1]
        vocab_logits = outputs.logits

        # 通常語彙の上位50のロジット平均
        top50 = vocab_logits[0, -1, :].topk(50).values.mean().item()
        vocab_top50_means.append(top50)

        # 仮想語彙のロジット
        virtual_logits = head.compute_virtual_logits(h)
        virtual_mean = virtual_logits[0, -1, :].mean().item()
        virtual_logit_means.append(virtual_mean)

    avg_vocab_top50 = sum(vocab_top50_means) / len(vocab_top50_means)
    avg_virtual = sum(virtual_logit_means) / len(virtual_logit_means)

    # 目標: 無関係な文脈での仮想語彙ロジット平均が
    # 通常語彙上位50の平均以下になるスケール係数
    if abs(avg_virtual) > 1e-6:
        scale = avg_vocab_top50 / avg_virtual
    else:
        scale = 1.0

    # スケール係数が 1.0 より大きい場合は補正不要（既に小さい）
    if scale >= 1.0:
        scale = 1.0

    calibrated_weights = head._U * scale

    return ContextualOutputHead(
        calibrated_weights,
        head.surface_names,
        head.layer_info + [{"calibration_scale": scale,
                            "avg_virtual_before": avg_virtual,
                            "avg_vocab_top50": avg_vocab_top50}],
    )


@torch.no_grad()
def build_contextual_output_head(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    registry: VocabularyRegistry,
    method: ContextualMethod = ContextualMethod.NORMED_HIDDEN,
    context_templates: dict[str, list[str]] | None = None,
) -> ContextualOutputHead:
    """レジストリの全仮想トークンに対して内部表現を抽出し、
    ContextualOutputHead を構築する。

    Args:
        model: 言語モデル。
        tokenizer: トークナイザ。
        registry: 仮想トークンのレジストリ。
        method: 内部表現抽出方式。
        context_templates: MULTI_CONTEXT で使うテンプレート辞書。
            ``None`` の場合はモジュール既定の ``_CONTEXT_TEMPLATES`` を使う。

    Returns:
        構築済みの ContextualOutputHead。
    """
    device = next(model.parameters()).device
    output_weight = model.get_output_embeddings().weight.detach()

    templates_dict = context_templates if context_templates is not None else _CONTEXT_TEMPLATES

    weight_rows = []
    surface_names = []
    layer_info_list = []

    for vtoken in registry:
        # 複合語フレーズを構成: 構成語をスペース区切りで結合
        phrase = " ".join(vtoken.components)
        surface_names.append(vtoken.surface)

        if method == ContextualMethod.LAST_HIDDEN:
            h = _extract_hidden_at_last_subword(
                model, tokenizer, phrase, layer_index=-1,
            )
            h = _norm_match_to_vocab(h, output_weight)
            weight_rows.append(h)
            layer_info_list.append({"method": "last_hidden", "phrase": phrase})

        elif method == ContextualMethod.NORMED_HIDDEN:
            h = _extract_with_norm(model, tokenizer, phrase)
            h = _norm_match_to_vocab(h, output_weight)
            weight_rows.append(h)
            layer_info_list.append({"method": "normed_hidden", "phrase": phrase})

        elif method == ContextualMethod.MULTI_CONTEXT:
            relation_key = vtoken.relation.value
            templates = templates_dict.get(relation_key, ["{phrase}"])
            hidden_states = []
            for tmpl in templates:
                ctx_text = tmpl.format(phrase=phrase)
                # フレーズの最終サブワード位置ではなく、
                # 文全体の最終位置を使う
                h = _extract_with_norm(model, tokenizer, ctx_text)
                hidden_states.append(h)
            h_avg = torch.stack(hidden_states).mean(dim=0)
            h_avg = _norm_match_to_vocab(h_avg, output_weight)
            weight_rows.append(h_avg)
            layer_info_list.append({
                "method": "multi_context",
                "phrase": phrase,
                "num_contexts": len(templates),
            })

        elif method == ContextualMethod.LAYERWISE_BEST:
            comp_ids = vtoken.component_token_ids
            h, best_layer, scores = _find_best_layer(
                model, tokenizer, phrase, comp_ids,
            )
            # 最適層の隠れ状態にノルム補正を適用
            h = _norm_match_to_vocab(h, output_weight)
            weight_rows.append(h)
            layer_info_list.append({
                "method": "layerwise_best",
                "phrase": phrase,
                "best_layer": best_layer,
                "layer_scores": scores,
            })

        else:
            raise ValueError(f"未対応の方式: {method}")

    output_weights = torch.stack(weight_rows).to(device)
    return ContextualOutputHead(output_weights, surface_names, layer_info_list)
