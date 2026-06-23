"""出力側の仮想logit追加。

通常の lm_head 出力に仮想語彙の logit を連結し、
新語彙を通常語彙と同じ softmax 候補として扱えるようにする。
"""

from __future__ import annotations

from enum import Enum

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


class AggregationMethod(Enum):
    """構成トークン logit の集約方式。"""

    FLAT_MEAN = "flat_mean"
    WORD_MEAN = "word_mean"
    WORD_MIN = "word_min"


class ComponentLogitHead:
    """構成トークンの logit 集約による文脈依存型仮想 logit 計算。

    通常の :class:`VirtualLogitHead` が固定出力重みベクトル w_z との
    内積で logit を計算するのに対し、本クラスはモデルの通常出力 logit
    から構成トークンの logit を読み取り、集約で仮想 logit を算出する。

    集約方式:

    - ``FLAT_MEAN``: 全サブワードの logit を平均する。
    - ``WORD_MEAN``: 語レベルの平均 logit を語間で平均する。
    - ``WORD_MIN``: 語レベルの平均 logit の最小値を取る。

    ``baseline`` を指定すると、各トークンの文脈非依存成分を差し引き、
    文脈依存の信号のみを集約する。

    Args:
        word_component_ids: 各仮想語彙の語レベル構成情報。
            ``word_component_ids[i][j]`` は仮想語彙 i の構成語 j の
            サブワードトークンID列。
        surface_names: 各仮想語彙の表示文字列。
        aggregation: 集約方式。
        baseline: 各トークンのベースライン logit (形状: ``[V]``)。
            ``None`` の場合は生の logit を使う。
        baseline_std: 各トークンの logit 標準偏差 (形状: ``[V]``)。
            ``baseline`` と同時に指定すると z 正規化を行う。
    """

    def __init__(
        self,
        word_component_ids: list[list[list[int]]],
        surface_names: list[str],
        aggregation: AggregationMethod = AggregationMethod.WORD_MIN,
        baseline: torch.Tensor | None = None,
        baseline_std: torch.Tensor | None = None,
    ) -> None:
        self._word_ids = word_component_ids
        self._names = surface_names
        self._agg = aggregation
        self._baseline = baseline
        self._baseline_std = baseline_std

        self._flat_ids = []
        for word_groups in word_component_ids:
            flat = []
            for wids in word_groups:
                flat.extend(wids)
            self._flat_ids.append(flat)

    @property
    def num_virtual(self) -> int:
        """仮想語彙数。"""
        return len(self._names)

    @property
    def surface_names(self) -> list[str]:
        """仮想語彙の表示文字列リスト。"""
        return list(self._names)

    def compute_virtual_logits(
        self, vocab_logits_last: torch.Tensor
    ) -> torch.Tensor:
        """通常語彙の logit ベクトルから仮想語彙の logit を計算する。

        Args:
            vocab_logits_last: 最終トークン位置の logit (形状: ``[V]``)。

        Returns:
            仮想語彙の logit (形状: ``[m]``)。
        """
        if self._baseline is not None and self._baseline_std is not None:
            dev = vocab_logits_last.device
            effective = (
                vocab_logits_last - self._baseline.to(dev)
            ) / (self._baseline_std.to(dev) + 1e-6)
        elif self._baseline is not None:
            effective = vocab_logits_last - self._baseline.to(
                vocab_logits_last.device
            )
        else:
            effective = vocab_logits_last

        scores = []
        for i, word_groups in enumerate(self._word_ids):
            if self._agg == AggregationMethod.FLAT_MEAN:
                ids = self._flat_ids[i]
                scores.append(effective[ids].mean())
            else:
                word_scores = []
                for wids in word_groups:
                    word_scores.append(
                        effective[wids].mean()
                    )
                ws = torch.stack(word_scores)
                if self._agg == AggregationMethod.WORD_MIN:
                    scores.append(ws.min())
                else:
                    scores.append(ws.mean())
        return torch.stack(scores)

    def get_virtual_rankings(
        self,
        hidden_state: torch.Tensor,
        vocab_logits: torch.Tensor,
    ) -> list[dict]:
        """各仮想語彙の拡張 logit 空間内での順位・確率を返す。

        Args:
            hidden_state: 未使用（互換性のため保持）。
            vocab_logits: 通常語彙の logit (形状: ``[1, seq, V]``)。

        Returns:
            各仮想語彙の logit 値・確率・順位を含む辞書のリスト。
        """
        last_logits = vocab_logits[0, -1, :]
        virtual_logits = self.compute_virtual_logits(last_logits)

        extended = torch.cat([last_logits, virtual_logits])
        probs = F.softmax(extended, dim=-1)

        sorted_indices = extended.argsort(descending=True)
        rank_map = torch.zeros_like(extended, dtype=torch.long)
        rank_map[sorted_indices] = torch.arange(
            len(sorted_indices), device=extended.device,
        )

        V = last_logits.shape[0]
        rankings = []
        for i, name in enumerate(self._names):
            idx = V + i
            rankings.append({
                "surface": name,
                "logit": extended[idx].item(),
                "probability": probs[idx].item(),
                "rank": rank_map[idx].item(),
                "total_candidates": len(extended),
            })
        return rankings


class LayerAggregation(Enum):
    """層間集約方式。"""

    SINGLE = "single"
    MEAN = "mean"
    MAX = "max"


class LayerwiseLogitHead:
    """中間層の logit lens 射影から構成トークン logit を読み出す。

    各層の隠れ状態を最終層正規化と出力重み行列で射影し、
    構成トークンの logit を層ごとに取得して集約する。
    最終層は次トークン予測に特化しているため既出語が抑制されるが、
    中間層ではこの抑制が弱く文脈の主題的情報が残る可能性がある。

    Args:
        output_weight: 出力重み行列 (形状: ``[V, d]``)。
        final_norm: 最終層正規化モジュール。``None`` で正規化を省略。
        word_component_ids: 各仮想語彙の語レベル構成情報。
        surface_names: 各仮想語彙の表示文字列。
        aggregation: 語内・語間の集約方式。
        layer_indices: 使用する層のインデックス。``None`` で全層。
        layer_aggregation: 層間の集約方式。
    """

    def __init__(
        self,
        output_weight: torch.Tensor,
        final_norm: torch.nn.Module | None,
        word_component_ids: list[list[list[int]]],
        surface_names: list[str],
        aggregation: AggregationMethod = AggregationMethod.WORD_MIN,
        layer_indices: list[int] | None = None,
        layer_aggregation: LayerAggregation = LayerAggregation.MEAN,
    ) -> None:
        self._W = output_weight
        self._norm = final_norm
        self._word_ids = word_component_ids
        self._names = surface_names
        self._agg = aggregation
        self._layer_indices = layer_indices
        self._layer_agg = layer_aggregation

        self._flat_ids: list[list[int]] = []
        for word_groups in word_component_ids:
            flat: list[int] = []
            for wids in word_groups:
                flat.extend(wids)
            self._flat_ids.append(flat)

    @property
    def num_virtual(self) -> int:
        """仮想語彙数。"""
        return len(self._names)

    @property
    def surface_names(self) -> list[str]:
        """仮想語彙の表示文字列リスト。"""
        return list(self._names)

    def _logit_lens(self, hidden_state_last: torch.Tensor) -> torch.Tensor:
        """隠れ状態に最終層正規化を適用し W で射影する。"""
        h = hidden_state_last
        if self._norm is not None:
            h = self._norm(h.unsqueeze(0)).squeeze(0)
        return h @ self._W.T

    def _virtual_scores_from_logits(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        """logit ベクトルから構成トークン logit を抽出し集約する。"""
        scores = []
        for i, word_groups in enumerate(self._word_ids):
            if self._agg == AggregationMethod.FLAT_MEAN:
                ids = self._flat_ids[i]
                scores.append(logits[ids].mean())
            else:
                word_scores = []
                for wids in word_groups:
                    word_scores.append(logits[wids].mean())
                ws = torch.stack(word_scores)
                if self._agg == AggregationMethod.WORD_MIN:
                    scores.append(ws.min())
                else:
                    scores.append(ws.mean())
        return torch.stack(scores)

    def get_virtual_rankings(
        self,
        all_hidden_states: tuple[torch.Tensor, ...],
        vocab_logits: torch.Tensor,
    ) -> list[dict]:
        """各仮想語彙の logit lens 由来スコアと順位を返す。

        SINGLE モードではその層の logit lens 全体を基準に順位付けする。
        MEAN/MAX モードでは最終層の通常 logit を基準とする。

        Args:
            all_hidden_states: 全層の隠れ状態タプル
                (形状: ``(num_layers+1, [1, seq, d])``)。
            vocab_logits: 通常語彙の logit (形状: ``[1, seq, V]``)。

        Returns:
            各仮想語彙の logit 値・順位を含む辞書のリスト。
        """
        layers = self._layer_indices
        if layers is None:
            layers = list(range(len(all_hidden_states)))

        if self._layer_agg == LayerAggregation.SINGLE:
            layer_idx = layers[0]
            h = all_hidden_states[layer_idx][0, -1, :]
            layer_logits = self._logit_lens(h)
            virtual_logits = self._virtual_scores_from_logits(layer_logits)
            base_logits = layer_logits
        else:
            layer_scores = []
            for layer_idx in layers:
                h = all_hidden_states[layer_idx][0, -1, :]
                logits_L = self._logit_lens(h)
                scores = self._virtual_scores_from_logits(logits_L)
                layer_scores.append(scores)
            stacked = torch.stack(layer_scores)

            if self._layer_agg == LayerAggregation.MAX:
                virtual_logits = stacked.max(dim=0).values
            else:
                virtual_logits = stacked.mean(dim=0)
            base_logits = vocab_logits[0, -1, :]

        extended = torch.cat([base_logits, virtual_logits])
        probs = F.softmax(extended, dim=-1)
        sorted_indices = extended.argsort(descending=True)
        rank_map = torch.zeros_like(extended, dtype=torch.long)
        rank_map[sorted_indices] = torch.arange(
            len(sorted_indices), device=extended.device,
        )

        V = base_logits.shape[0]
        rankings = []
        for i, name in enumerate(self._names):
            idx = V + i
            rankings.append({
                "surface": name,
                "logit": extended[idx].item(),
                "probability": probs[idx].item(),
                "rank": rank_map[idx].item(),
                "total_candidates": len(extended),
            })
        return rankings


class GateMethod(Enum):
    """隠れ状態ゲートの集約方式。"""

    WORD_MEAN = "gate_word_mean"
    WORD_MIN = "gate_word_min"
    WORD_PRODUCT = "gate_word_product"


class HiddenStateGate:
    """隠れ状態と構成語埋め込みの類似度に基づく仮想 logit 計算。

    最終隠れ状態 h が構成語の概念をどの程度符号化しているかを
    入力埋め込みとの余弦類似度で測定し、仮想語彙のスコアとする。

    次トークン予測 logit に依存しないため、構成語がプロンプト中に
    既出であっても文脈弁別が可能。

    Args:
        word_component_embeddings: 各仮想語彙の語レベル構成埋め込み。
            ``word_component_embeddings[i][j]`` は仮想語彙 i の
            構成語 j の平均入力埋め込み (形状: ``[d]``)。
        surface_names: 各仮想語彙の表示文字列。
        gate_method: 語間の集約方式。
    """

    def __init__(
        self,
        word_component_embeddings: list[list[torch.Tensor]],
        surface_names: list[str],
        gate_method: GateMethod = GateMethod.WORD_MIN,
    ) -> None:
        self._word_embs = word_component_embeddings
        self._names = surface_names
        self._gate = gate_method

    @property
    def num_virtual(self) -> int:
        """仮想語彙数。"""
        return len(self._names)

    @property
    def surface_names(self) -> list[str]:
        """仮想語彙の表示文字列リスト。"""
        return list(self._names)

    def compute_virtual_logits(
        self, hidden_state_last: torch.Tensor
    ) -> torch.Tensor:
        """最終隠れ状態から仮想語彙のスコアを計算する。

        Args:
            hidden_state_last: 最終トークン位置の隠れ状態 (形状: ``[d]``)。

        Returns:
            仮想語彙のスコア (形状: ``[m]``)。
        """
        h = hidden_state_last
        scores = []
        for word_embs in self._word_embs:
            word_sims = []
            for emb in word_embs:
                sim = F.cosine_similarity(
                    h.unsqueeze(0),
                    emb.unsqueeze(0).to(h.device),
                )
                word_sims.append(sim.squeeze())
            ws = torch.stack(word_sims)

            if self._gate == GateMethod.WORD_MIN:
                scores.append(ws.min())
            elif self._gate == GateMethod.WORD_PRODUCT:
                scores.append(ws.prod())
            else:
                scores.append(ws.mean())
        return torch.stack(scores)

    def get_virtual_rankings(
        self,
        hidden_state: torch.Tensor,
        vocab_logits: torch.Tensor,
    ) -> list[dict]:
        """各仮想語彙の拡張 logit 空間内での順位・確率を返す。

        仮想語彙のスコアは余弦類似度（-1〜1）であり、通常 logit と
        スケールが異なる。logit 空間での順位を計算するため、
        通常語彙の logit 範囲にリスケーリングする。

        Args:
            hidden_state: 最終層の隠れ状態 (形状: ``[1, seq, d]``)。
            vocab_logits: 通常語彙の logit (形状: ``[1, seq, V]``)。

        Returns:
            各仮想語彙の logit 値・確率・順位を含む辞書のリスト。
        """
        h = hidden_state[0, -1, :]
        last_logits = vocab_logits[0, -1, :]
        virtual_scores = self.compute_virtual_logits(h)

        vocab_mean = last_logits.mean()
        vocab_std = last_logits.std()
        virtual_logits = vocab_mean + virtual_scores * vocab_std

        extended = torch.cat([last_logits, virtual_logits])
        probs = F.softmax(extended, dim=-1)

        sorted_indices = extended.argsort(descending=True)
        rank_map = torch.zeros_like(extended, dtype=torch.long)
        rank_map[sorted_indices] = torch.arange(
            len(sorted_indices), device=extended.device,
        )

        V = last_logits.shape[0]
        rankings = []
        for i, name in enumerate(self._names):
            idx = V + i
            rankings.append({
                "surface": name,
                "logit": extended[idx].item(),
                "probability": probs[idx].item(),
                "rank": rank_map[idx].item(),
                "total_candidates": len(extended),
            })
        return rankings


def build_hidden_state_gate(
    tokenizer: PreTrainedTokenizerBase,
    registry: VocabularyRegistry,
    embedding_weight: torch.Tensor,
    gate_method: GateMethod = GateMethod.WORD_MIN,
) -> HiddenStateGate:
    """トークナイザとレジストリから HiddenStateGate を構築する。

    Args:
        tokenizer: トークナイザ。
        registry: 仮想トークンのレジストリ。
        embedding_weight: 入力埋め込み行列 (形状: ``[V, d]``)。
        gate_method: 語間の集約方式。

    Returns:
        構築済みの :class:`HiddenStateGate`。
    """
    word_embs_all = []
    surface_names = []
    for vtoken in registry:
        word_embs = []
        for word in vtoken.components:
            ids = tokenizer.encode(word, add_special_tokens=False)
            emb = embedding_weight[ids].mean(dim=0)
            word_embs.append(emb)
        word_embs_all.append(word_embs)
        surface_names.append(vtoken.surface)
    return HiddenStateGate(word_embs_all, surface_names, gate_method)


@torch.no_grad()
def compute_logit_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
) -> torch.Tensor:
    """多様なプロンプトから各トークンのベースライン logit を推定する。

    各プロンプトの最終位置の logit ベクトルを収集し、
    トークンごとに平均して文脈非依存成分を得る。

    Args:
        model: 言語モデル。
        tokenizer: トークナイザ。
        prompts: ベースライン推定に使うプロンプト集合。

    Returns:
        ベースライン logit ベクトル (形状: ``[V]``)。
    """
    device = next(model.parameters()).device
    logit_sum = None
    count = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        out = model(input_ids=input_ids)
        last_logits = out.logits[0, -1, :].float()

        if logit_sum is None:
            logit_sum = torch.zeros_like(last_logits)
        logit_sum += last_logits
        count += 1

    return logit_sum / count


@torch.no_grad()
def compute_logit_statistics(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """各トークンの logit の平均と標準偏差を推定する。

    Args:
        model: 言語モデル。
        tokenizer: トークナイザ。
        prompts: 推定に使うプロンプト集合。

    Returns:
        (平均, 標準偏差) のタプル。各 ``[V]`` 形状。
    """
    device = next(model.parameters()).device
    all_logits = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        out = model(input_ids=input_ids)
        all_logits.append(out.logits[0, -1, :].float().cpu())

    stacked = torch.stack(all_logits)
    return stacked.mean(dim=0), stacked.std(dim=0)


BASELINE_PROMPTS = [
    "The",
    "In the",
    "It is",
    "A popular",
    "The most",
    "One of the",
    "This is a",
    "There are many",
    "People often",
    "According to",
    "The first",
    "In recent years",
    "Scientists have",
    "The government",
    "A new study",
    "The city of",
    "During the",
    "Many people",
    "The company",
    "In addition to",
    "The history of",
    "While some",
    "The weather",
    "A famous",
    "The price of",
    "In the morning",
    "The best",
    "After the",
    "The food",
    "Music is",
    "The building",
    "Books about",
    "The river",
    "Technology has",
    "The old",
    "Cooking with",
    "The school",
    "Sports and",
    "The country",
    "Animals in",
    "The mountain",
    "Art and",
    "The road",
    "Travel to",
    "The ocean",
    "Children love",
    "The garden",
    "Language is",
    "The night",
    "Flowers in",
    "The bridge",
    "Health and",
    "The market",
    "Games for",
    "The forest",
    "Water and",
    "The island",
    "The museum",
    "Colors of",
    "The castle",
    "Films about",
    "The kitchen",
    "Dreams of",
    "The library",
    "Peace and",
    "The harbor",
    "Stars and",
    "The village",
    "Wind and",
    "The clock",
    "Snow in",
    "The train",
    "Fire and",
    "The park",
    "Gold and",
    "The tower",
    "Stones from",
    "The window",
    "Ice and",
    "The door",
    "Silk from",
    "The ship",
    "Iron and",
    "The bell",
    "Rain in",
    "The wall",
    "Sand and",
    "The well",
    "Light and",
    "The path",
    "Wood from",
    "The ring",
    "Salt and",
    "The coin",
    "Glass and",
    "The crown",
    "Earth and",
    "The sword",
    "Steel and",
    "The drum",
    "Clay from",
]


def build_layerwise_logit_head(
    tokenizer: PreTrainedTokenizerBase,
    registry: VocabularyRegistry,
    output_weight: torch.Tensor,
    final_norm: torch.nn.Module | None,
    aggregation: AggregationMethod = AggregationMethod.WORD_MIN,
    layer_indices: list[int] | None = None,
    layer_aggregation: LayerAggregation = LayerAggregation.MEAN,
) -> LayerwiseLogitHead:
    """トークナイザとレジストリから LayerwiseLogitHead を構築する。

    Args:
        tokenizer: トークナイザ。
        registry: 仮想トークンのレジストリ。
        output_weight: 出力重み行列 (形状: ``[V, d]``)。
        final_norm: 最終層正規化モジュール。
        aggregation: 語内・語間の集約方式。
        layer_indices: 使用する層のインデックス。
        layer_aggregation: 層間の集約方式。

    Returns:
        構築済みの :class:`LayerwiseLogitHead`。
    """
    word_component_ids = []
    surface_names = []
    for vtoken in registry:
        word_groups = []
        for word in vtoken.components:
            ids = tokenizer.encode(word, add_special_tokens=False)
            word_groups.append(ids)
        word_component_ids.append(word_groups)
        surface_names.append(vtoken.surface)
    return LayerwiseLogitHead(
        output_weight, final_norm,
        word_component_ids, surface_names,
        aggregation, layer_indices, layer_aggregation,
    )


def build_component_logit_head(
    tokenizer: PreTrainedTokenizerBase,
    registry: VocabularyRegistry,
    aggregation: AggregationMethod = AggregationMethod.WORD_MIN,
    baseline: torch.Tensor | None = None,
    baseline_std: torch.Tensor | None = None,
) -> ComponentLogitHead:
    """トークナイザとレジストリから ComponentLogitHead を構築する。

    Args:
        tokenizer: トークナイザ。
        registry: 仮想トークンのレジストリ。
        aggregation: 集約方式。
        baseline: 各トークンのベースライン logit。
            ``None`` で生の logit を使う。
        baseline_std: 各トークンの logit 標準偏差。
            ``baseline`` と同時指定で z 正規化。

    Returns:
        構築済みの :class:`ComponentLogitHead`。
    """
    word_component_ids = []
    surface_names = []
    for vtoken in registry:
        word_groups = []
        for word in vtoken.components:
            ids = tokenizer.encode(word, add_special_tokens=False)
            word_groups.append(ids)
        word_component_ids.append(word_groups)
        surface_names.append(vtoken.surface)
    return ComponentLogitHead(
        word_component_ids, surface_names, aggregation,
        baseline, baseline_std,
    )


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
