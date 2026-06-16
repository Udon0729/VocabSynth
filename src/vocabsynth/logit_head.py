"""出力側の仮想logit追加（スタブ）。

第三段階で本実装する。現段階ではインターフェースのみ定義する。
"""

from __future__ import annotations

import torch


class VirtualLogitHead:
    """通常のlm_head出力に仮想語彙のlogitを追加する。

    通常語彙の logit ベクトル ``l_vocab`` (形状: ``[V]``) に対して、
    仮想語彙の logit ``l_virtual`` (形状: ``[m]``) を連結し、
    ``[V + m]`` 次元の候補ベクトルを構成する。

    Args:
        output_weights: 仮想語彙の出力重み行列 ``U_Z`` (形状: ``[m, d]``)。
        surface_names: 各仮想語彙の表示文字列。
    """

    def __init__(
        self,
        output_weights: torch.Tensor,
        surface_names: list[str],
    ) -> None:
        self._U = output_weights
        self._names = surface_names

    def extend_logits(
        self, hidden_state: torch.Tensor, vocab_logits: torch.Tensor
    ) -> torch.Tensor:
        """隠れ状態から仮想語彙logitを計算し、通常logitと連結する。

        Args:
            hidden_state: 最終層の隠れ状態 (形状: ``[batch, seq, d]``)。
            vocab_logits: 通常語彙のlogit (形状: ``[batch, seq, V]``)。

        Returns:
            拡張logit (形状: ``[batch, seq, V + m]``)。
        """
        raise NotImplementedError("第三段階で実装予定")
