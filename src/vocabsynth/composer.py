"""構成トークンからの埋め込み合成。

構成トークンID列と入力埋め込み行列から、新語彙の合成埋め込み E_z を生成する。
合成方式として単純平均、後部主要部重み、トークン長重みの三種を提供する。
"""

from __future__ import annotations

from enum import Enum

import torch
from transformers import PreTrainedTokenizerBase


class ComposeMethod(Enum):
    """埋め込み合成方式。"""

    MEAN = "mean"
    HEAD_WEIGHTED = "head_weighted"
    LENGTH_WEIGHTED = "length_weighted"


class EmbeddingComposer:
    """構成トークン埋め込みから新語彙埋め込みを合成する。

    Args:
        embedding_weight: 入力埋め込み行列 ``E`` (形状: ``[V, d]``)。
        tokenizer: 構成トークンの文字列長を取得するためのトークナイザ。
            :attr:`ComposeMethod.LENGTH_WEIGHTED` を使う場合に必要。
        head_weight: :attr:`ComposeMethod.HEAD_WEIGHTED` で末尾要素に
            割り当てる重み。先頭要素には ``1 - head_weight`` を均等配分する。
    """

    def __init__(
        self,
        embedding_weight: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase | None = None,
        head_weight: float = 0.6,
    ) -> None:
        self._E = embedding_weight
        self._tokenizer = tokenizer
        self._head_weight = head_weight

    def compose(
        self,
        token_ids: list[int],
        method: ComposeMethod = ComposeMethod.MEAN,
        normalize: bool = True,
    ) -> torch.Tensor:
        """構成トークンID列から合成埋め込みを生成する。

        Args:
            token_ids: 構成トークンのID列。
            method: 合成方式。
            normalize: ``True`` の場合、合成ベクトルのノルムを
                構成トークンの平均ノルムに合わせる。

        Returns:
            合成埋め込みベクトル (形状: ``[d]``)。
        """
        embeddings = self._E[token_ids]
        weights = self._compute_weights(token_ids, method)
        weights = weights.to(embeddings.device)

        composed = (weights.unsqueeze(1) * embeddings).sum(dim=0)

        if normalize:
            target_norm = embeddings.norm(dim=1).mean()
            current_norm = composed.norm()
            if current_norm > 0:
                composed = composed * (target_norm / current_norm)

        return composed

    def compose_batch(
        self,
        token_id_lists: list[list[int]],
        method: ComposeMethod = ComposeMethod.MEAN,
        normalize: bool = True,
    ) -> torch.Tensor:
        """複数の新語彙に対して一括で合成埋め込みを生成する。

        Args:
            token_id_lists: 各新語彙の構成トークンID列のリスト。
            method: 合成方式。
            normalize: ノルム補正の有無。

        Returns:
            合成埋め込み行列 (形状: ``[m, d]``)。
        """
        return torch.stack([
            self.compose(ids, method, normalize)
            for ids in token_id_lists
        ])

    def compose_random(self, token_ids: list[int]) -> torch.Tensor:
        """ランダム初期化による埋め込みを生成する（ベースライン用）。

        ノルムは構成トークンの平均ノルムに合わせる。

        Args:
            token_ids: 構成トークンのID列（ノルム参照用）。

        Returns:
            ランダム埋め込みベクトル (形状: ``[d]``)。
        """
        d = self._E.shape[1]
        random_vec = torch.randn(d, device=self._E.device, dtype=self._E.dtype)
        target_norm = self._E[token_ids].norm(dim=1).mean()
        return random_vec * (target_norm / random_vec.norm())

    def _compute_weights(
        self, token_ids: list[int], method: ComposeMethod
    ) -> torch.Tensor:
        k = len(token_ids)
        if k == 1:
            return torch.ones(1)

        if method == ComposeMethod.MEAN:
            return torch.ones(k) / k

        if method == ComposeMethod.HEAD_WEIGHTED:
            weights = torch.full((k,), (1.0 - self._head_weight) / (k - 1))
            weights[-1] = self._head_weight
            return weights

        if method == ComposeMethod.LENGTH_WEIGHTED:
            if self._tokenizer is None:
                raise ValueError(
                    "LENGTH_WEIGHTED にはトークナイザの指定が必要"
                )
            lengths = torch.tensor([
                len(self._tokenizer.decode([tid]).strip())
                for tid in token_ids
            ], dtype=torch.float)
            lengths = lengths.clamp(min=1.0)
            return lengths / lengths.sum()

        raise ValueError(f"未対応の合成方式: {method}")
