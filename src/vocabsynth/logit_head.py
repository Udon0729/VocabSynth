"""出力側の仮想logit追加。

通常の lm_head 出力に仮想語彙の logit を連結し、
新語彙を通常語彙と同じ softmax 候補として扱えるようにする。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from vocabsynth.composer import ComposeMethod, EmbeddingComposer
from vocabsynth.registry import VocabularyRegistry


class VirtualLogitHead:
    """通常の lm_head 出力に仮想語彙の logit を追加する。

    通常語彙の logit ベクトル ``l_vocab`` (形状: ``[V]``) に対して、
    仮想語彙の logit ``l_virtual`` (形状: ``[m]``) を連結し、
    ``[V + m]`` 次元の候補ベクトルを構成する。

    出力重み ``U_z`` の構成方法は二通りある:

    - weight tying モデル: 入力埋め込みと出力重みが共有されるため、
      合成入力埋め込みをそのまま出力重みとして使う。
    - 非 weight tying モデル: lm_head の重み行列から
      構成トークン行を取得し、別途合成する。

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

    @property
    def num_virtual(self) -> int:
        """仮想語彙数。"""
        return self._U.shape[0]

    @property
    def surface_names(self) -> list[str]:
        """仮想語彙の表示文字列リスト。"""
        return list(self._names)

    def compute_virtual_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """隠れ状態から仮想語彙の logit のみを計算する。

        Args:
            hidden_state: 最終層の隠れ状態 (形状: ``[batch, seq, d]``)。

        Returns:
            仮想語彙の logit (形状: ``[batch, seq, m]``)。
        """
        return torch.matmul(hidden_state, self._U.T)

    def extend_logits(
        self, hidden_state: torch.Tensor, vocab_logits: torch.Tensor
    ) -> torch.Tensor:
        """隠れ状態から仮想語彙 logit を計算し、通常 logit と連結する。

        Args:
            hidden_state: 最終層の隠れ状態 (形状: ``[batch, seq, d]``)。
            vocab_logits: 通常語彙の logit (形状: ``[batch, seq, V]``)。

        Returns:
            拡張 logit (形状: ``[batch, seq, V + m]``)。
        """
        virtual_logits = self.compute_virtual_logits(hidden_state)
        return torch.cat([vocab_logits, virtual_logits], dim=-1)

    def get_virtual_rankings(
        self,
        hidden_state: torch.Tensor,
        vocab_logits: torch.Tensor,
    ) -> list[dict]:
        """各仮想語彙の拡張 logit 空間内での順位・確率を返す。

        最終トークン位置のみを対象とする。

        Args:
            hidden_state: 最終層の隠れ状態 (形状: ``[1, seq, d]``)。
            vocab_logits: 通常語彙の logit (形状: ``[1, seq, V]``)。

        Returns:
            各仮想語彙について、logit 値・確率・順位を含む辞書のリスト。
        """
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


def build_virtual_logit_head(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    registry: VocabularyRegistry,
    composer: EmbeddingComposer,
    method: ComposeMethod = ComposeMethod.MEAN,
    use_lm_head_weights: bool = False,
) -> VirtualLogitHead:
    """モデルとレジストリから VirtualLogitHead を構築する。

    Weight tying モデルでは入力埋め込みの合成結果を出力重みに流用する。
    非 weight tying モデルでは lm_head の重み行列から別途合成する。

    Args:
        model: 言語モデル。
        tokenizer: トークナイザ。
        registry: 仮想トークンのレジストリ。
        composer: 埋め込み合成器（入力側用）。
        method: 合成方式。
        use_lm_head_weights: ``True`` の場合、入力埋め込みではなく
            lm_head の重みから出力重みを合成する。weight tying でない
            モデルで使う。

    Returns:
        構築済みの :class:`VirtualLogitHead`。
    """
    surfaces = []
    weight_rows = []

    if use_lm_head_weights:
        output_layer = model.get_output_embeddings()
        lm_head_weight = output_layer.weight.detach()
        lm_composer = EmbeddingComposer(lm_head_weight, tokenizer)
    else:
        lm_composer = composer

    for vtoken in registry:
        surfaces.append(vtoken.surface)
        w = lm_composer.compose(vtoken.component_token_ids, method)
        weight_rows.append(w)

    output_weights = torch.stack(weight_rows).to(
        next(model.parameters()).device
    )
    return VirtualLogitHead(output_weights, surfaces)
