"""関係補正ベクトルの推定と適用。

既存の複合語参照ペアから構成要素間の関係方向を抽出し、
新語彙の合成埋め込みに非学習的な補正を加える。

concept.md の実装選択肢2（文脈的表現抽出）を採用する。
複合語フレーズをモデルに通し、最終層隠れ状態の平均プーリングで
複合語表現を得る。この表現と構成要素の静的ベース合成との差分を
関係補正ベクトルとする。

.. note::
    隠れ状態と静的埋め込みは異なるベクトル空間にある。
    補正ベクトルの適用時にノルム補正を行うことで、
    スケールの不整合を緩和する。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from vocabsynth.composer import ComposeMethod, EmbeddingComposer
from vocabsynth.registry import RelationType


@dataclass
class ReferencePair:
    """関係補正の参照ペア。

    Attributes:
        modifier: 修飾要素（例: ``"New York"``、``"Swiss"``）。
        head: 主要部（例: ``"pizza"``、``"cheese"``）。
        compound_phrase: 複合語フレーズ（例: ``"New York pizza"``）。
        relation: 構成要素間の意味関係。
    """

    modifier: str
    head: str
    compound_phrase: str
    relation: RelationType


DEFAULT_PAIRS: dict[RelationType, list[ReferencePair]] = {
    RelationType.PLACE_FOOD: [
        ReferencePair("New York", "pizza", "New York pizza", RelationType.PLACE_FOOD),
        ReferencePair("Chicago", "pizza", "Chicago pizza", RelationType.PLACE_FOOD),
        ReferencePair("Nashville", "chicken", "Nashville chicken", RelationType.PLACE_FOOD),
        ReferencePair("Hawaiian", "pizza", "Hawaiian pizza", RelationType.PLACE_FOOD),
        ReferencePair("Swiss", "cheese", "Swiss cheese", RelationType.PLACE_FOOD),
        ReferencePair("French", "bread", "French bread", RelationType.PLACE_FOOD),
        ReferencePair("Belgian", "chocolate", "Belgian chocolate", RelationType.PLACE_FOOD),
        ReferencePair("Turkish", "coffee", "Turkish coffee", RelationType.PLACE_FOOD),
        ReferencePair("Japanese", "ramen", "Japanese ramen", RelationType.PLACE_FOOD),
        ReferencePair("Indian", "curry", "Indian curry", RelationType.PLACE_FOOD),
    ],
    RelationType.PLACE_STRUCTURE: [
        ReferencePair("London", "Bridge", "London Bridge", RelationType.PLACE_STRUCTURE),
        ReferencePair("Brooklyn", "Bridge", "Brooklyn Bridge", RelationType.PLACE_STRUCTURE),
        ReferencePair("Tokyo", "Tower", "Tokyo Tower", RelationType.PLACE_STRUCTURE),
        ReferencePair("Eiffel", "Tower", "Eiffel Tower", RelationType.PLACE_STRUCTURE),
        ReferencePair("Berlin", "Wall", "Berlin Wall", RelationType.PLACE_STRUCTURE),
        ReferencePair("Great", "Wall", "Great Wall", RelationType.PLACE_STRUCTURE),
    ],
}


class RelationCorrector:
    """既存の参照ペアから関係補正ベクトルを推定・適用する。

    Args:
        model: 文脈的表現の抽出に使う言語モデル。
        tokenizer: トークナイザ。
        embedding_weight: 入力埋め込み行列 (形状: ``[V, d]``)。
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        embedding_weight: torch.Tensor,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._E = embedding_weight
        self._device = next(model.parameters()).device
        self._correction_cache: dict[tuple[RelationType, str], torch.Tensor] = {}

    @torch.no_grad()
    def compute_correction_vector(
        self,
        pairs: list[ReferencePair],
        base_method: ComposeMethod = ComposeMethod.MEAN,
    ) -> torch.Tensor:
        """参照ペア集合から関係補正ベクトル R_r を推定する。

        各参照ペアについて、複合語の文脈的表現とベース合成の差分を
        計算し、全ペアの平均を関係補正ベクトルとする。

        Args:
            pairs: 参照ペアのリスト。
            base_method: ベース合成方式。

        Returns:
            関係補正ベクトル R_r (形状: ``[d]``)。
        """
        deltas = []
        for pair in pairs:
            compound_repr = self._get_contextual_repr(pair.compound_phrase)
            base = self._compose_components(
                pair.modifier, pair.head, base_method
            )
            delta = compound_repr - base
            deltas.append(delta)

        R_r = torch.stack(deltas).mean(dim=0)
        return R_r

    def get_correction_vector(
        self,
        relation: RelationType,
        base_method: ComposeMethod = ComposeMethod.MEAN,
        pairs: list[ReferencePair] | None = None,
    ) -> torch.Tensor:
        """関係タイプに対する補正ベクトルを取得する（キャッシュ付き）。

        Args:
            relation: 関係タイプ。
            base_method: ベース合成方式。
            pairs: 参照ペア。``None`` の場合は既定のペアを使う。

        Returns:
            関係補正ベクトル R_r。
        """
        cache_key = (relation, base_method.value)
        if cache_key in self._correction_cache:
            return self._correction_cache[cache_key]

        if pairs is None:
            pairs = DEFAULT_PAIRS.get(relation, [])
        if not pairs:
            return torch.zeros(self._E.shape[1], device=self._device)

        R_r = self.compute_correction_vector(pairs, base_method)
        self._correction_cache[cache_key] = R_r
        return R_r

    def apply_correction(
        self,
        base_embedding: torch.Tensor,
        relation: RelationType,
        lambda_: float,
        base_method: ComposeMethod = ComposeMethod.MEAN,
        pairs: list[ReferencePair] | None = None,
    ) -> torch.Tensor:
        """ベース合成埋め込みに関係補正を適用する。

        R_r を単位ベクトルに正規化したうえで、ベース埋め込みの
        ノルムを基準にスケーリングする。

        .. math::
            E_z = \\text{normalize}\\bigl(
                E_{\\text{base}} + \\lambda \\cdot \\|E_{\\text{base}}\\|
                \\cdot \\hat{R}_r
            \\bigr)

        これにより λ は「ベース埋め込みノルムの何倍の補正を加えるか」
        を直接制御する。

        Args:
            base_embedding: ベース合成埋め込み (形状: ``[d]``)。
            relation: 関係タイプ。
            lambda_: 補正強度。``0`` で補正なし。
            base_method: ベース合成に使った方式（キャッシュキー用）。
            pairs: 参照ペア。``None`` で既定値。

        Returns:
            補正後の埋め込み (形状: ``[d]``)。
        """
        if lambda_ == 0.0:
            return base_embedding

        R_r = self.get_correction_vector(relation, base_method, pairs)
        base_norm = base_embedding.norm()

        R_r_unit = R_r / R_r.norm().clamp(min=1e-8)
        corrected = base_embedding + lambda_ * base_norm * R_r_unit

        if corrected.norm() > 0:
            corrected = corrected * (base_norm / corrected.norm())

        return corrected

    @torch.no_grad()
    def _get_contextual_repr(self, phrase: str) -> torch.Tensor:
        """フレーズの文脈的表現を取得する。

        フレーズをモデルに通し、最終層隠れ状態の平均プーリングで
        表現ベクトルを得る。
        """
        inputs = self._tokenizer(phrase, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._device)

        outputs = self._model(input_ids=input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0]
        phrase_repr = hidden.mean(dim=0)

        return phrase_repr

    def _compose_components(
        self,
        modifier: str,
        head: str,
        method: ComposeMethod,
    ) -> torch.Tensor:
        """修飾要素と主要部からベース合成を計算する。"""
        mod_ids = self._tokenizer.encode(modifier, add_special_tokens=False)
        head_ids = self._tokenizer.encode(head, add_special_tokens=False)

        mod_emb = self._E[mod_ids].mean(dim=0)
        head_emb = self._E[head_ids].mean(dim=0)

        if method == ComposeMethod.MEAN:
            return (mod_emb + head_emb) / 2
        elif method == ComposeMethod.HEAD_WEIGHTED:
            return 0.4 * mod_emb + 0.6 * head_emb
        elif method == ComposeMethod.LENGTH_WEIGHTED:
            mod_len = len(modifier)
            head_len = len(head)
            total = mod_len + head_len
            return (mod_len / total) * mod_emb + (head_len / total) * head_emb
        else:
            return (mod_emb + head_emb) / 2

    def diagnose(
        self,
        relation: RelationType,
        base_method: ComposeMethod = ComposeMethod.MEAN,
    ) -> dict:
        """補正ベクトルの診断情報を返す。

        各参照ペアの差分ベクトルのノルムとペア間余弦類似度を報告する。

        Args:
            relation: 関係タイプ。
            base_method: ベース合成方式。

        Returns:
            差分統計を含む辞書。
        """
        pairs = DEFAULT_PAIRS.get(relation, [])
        if not pairs:
            return {"error": "参照ペアなし"}

        deltas = []
        pair_info = []
        for pair in pairs:
            compound_repr = self._get_contextual_repr(pair.compound_phrase)
            base = self._compose_components(
                pair.modifier, pair.head, base_method
            )
            delta = compound_repr - base
            deltas.append(delta)
            pair_info.append({
                "compound": pair.compound_phrase,
                "delta_norm": delta.norm().item(),
                "compound_repr_norm": compound_repr.norm().item(),
                "base_norm": base.norm().item(),
            })

        stacked = torch.stack(deltas)
        R_r = stacked.mean(dim=0)

        cos_sims = []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                cos = torch.nn.functional.cosine_similarity(
                    deltas[i].unsqueeze(0), deltas[j].unsqueeze(0)
                ).item()
                cos_sims.append(cos)

        return {
            "relation": relation.value,
            "num_pairs": len(pairs),
            "R_r_norm": R_r.norm().item(),
            "mean_delta_norm": stacked.norm(dim=1).mean().item(),
            "std_delta_norm": stacked.norm(dim=1).std().item(),
            "mean_pairwise_cosine": sum(cos_sims) / max(len(cos_sims), 1),
            "pair_details": pair_info,
        }
