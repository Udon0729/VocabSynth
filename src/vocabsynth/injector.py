"""合成埋め込みの注入。

テキスト中の新語彙出現位置を検出し、通常の埋め込み列に合成埋め込みを
挿入して ``inputs_embeds`` を構成する。トークナイザの語彙拡張は行わない。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from vocabsynth.registry import VocabularyRegistry


@dataclass
class InjectionResult:
    """埋め込み注入の結果。

    Attributes:
        inputs_embeds: 合成埋め込みが挿入された埋め込み列
            (形状: ``[1, seq_len, d]``)。
        attention_mask: 注意マスク (形状: ``[1, seq_len]``)。
        virtual_positions: 仮想トークンが配置された位置のリスト。
            各要素は ``(開始位置, 終了位置, 表層文字列)`` のタプル。
    """

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    virtual_positions: list[tuple[int, int, str]]


class VirtualInputInjector:
    """テキスト中の仮想トークンを合成埋め込みに置換する。

    処理の流れ:

    1. テキスト中から登録済み仮想トークンの出現位置を検出する。
    2. 仮想トークンを除いたテキストをトークナイズする。
    3. 仮想トークン位置に合成埋め込みを挿入する。

    Args:
        model: 入力埋め込み層を持つ言語モデル。
        tokenizer: トークナイザ（語彙追加前）。
        registry: 仮想トークンのレジストリ。
        virtual_embeddings: 表層文字列をキーとする合成埋め込み辞書。
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        registry: VocabularyRegistry,
        virtual_embeddings: dict[str, torch.Tensor],
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._registry = registry
        self._virtual_embeddings = virtual_embeddings

    def inject(self, text: str) -> InjectionResult:
        """テキストを処理し、仮想トークン位置に合成埋め込みを注入する。

        Args:
            text: 入力テキスト。仮想トークンの表層文字列を含む。

        Returns:
            合成埋め込みが注入された :class:`InjectionResult`。
        """
        spans = self._find_virtual_spans(text)

        if not spans:
            inputs = self._tokenizer(text, return_tensors="pt")
            embed_layer = self._model.get_input_embeddings()
            device = next(self._model.parameters()).device
            input_ids = inputs["input_ids"].to(device)
            embeds = embed_layer(input_ids)
            return InjectionResult(
                inputs_embeds=embeds,
                attention_mask=inputs["attention_mask"].to(device),
                virtual_positions=[],
            )

        return self._build_with_virtual(text, spans)

    def _find_virtual_spans(self, text: str) -> list[tuple[int, int, str]]:
        """テキスト中の仮想トークン出現位置を検出する。

        Returns:
            ``(開始文字位置, 終了文字位置, 表層文字列)`` のリスト。
            出現位置の早い順にソートされている。
        """
        spans = []
        for surface in self._registry.surfaces():
            start = 0
            while True:
                idx = text.find(surface, start)
                if idx == -1:
                    break
                spans.append((idx, idx + len(surface), surface))
                start = idx + len(surface)
        spans.sort(key=lambda s: s[0])
        return spans

    def _build_with_virtual(
        self, text: str, spans: list[tuple[int, int, str]]
    ) -> InjectionResult:
        """仮想トークン位置を含むテキストから埋め込み列を構築する。"""
        embed_layer = self._model.get_input_embeddings()
        device = next(self._model.parameters()).device
        segments: list[torch.Tensor] = []
        virtual_positions: list[tuple[int, int, str]] = []
        current_pos = 0
        token_offset = 0

        prev_end = 0
        for char_start, char_end, surface in spans:
            prefix = text[prev_end:char_start]
            if prefix:
                prefix_ids = self._tokenizer.encode(
                    prefix, add_special_tokens=False
                )
                if prefix_ids:
                    prefix_embeds = embed_layer(
                        torch.tensor([prefix_ids], device=device)
                    ).squeeze(0)
                    segments.append(prefix_embeds)
                    token_offset += len(prefix_ids)

            v_embed = self._virtual_embeddings[surface].to(device)
            segments.append(v_embed.unsqueeze(0))
            virtual_positions.append(
                (token_offset, token_offset + 1, surface)
            )
            token_offset += 1
            prev_end = char_end

        suffix = text[prev_end:]
        if suffix:
            suffix_ids = self._tokenizer.encode(
                suffix, add_special_tokens=False
            )
            if suffix_ids:
                suffix_embeds = embed_layer(
                    torch.tensor([suffix_ids], device=device)
                ).squeeze(0)
                segments.append(suffix_embeds)
                token_offset += len(suffix_ids)

        all_embeds = torch.cat(segments, dim=0).unsqueeze(0)
        attention_mask = torch.ones(
            1, all_embeds.shape[1], dtype=torch.long, device=device
        )

        return InjectionResult(
            inputs_embeds=all_embeds,
            attention_mask=attention_mask,
            virtual_positions=virtual_positions,
        )


def build_multi_token_inputs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """仮想トークンなしの通常入力を構築する（ベースライン用）。

    Args:
        model: 言語モデル。
        tokenizer: トークナイザ。
        text: 入力テキスト。

    Returns:
        ``(inputs_embeds, attention_mask)`` のタプル。
    """
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(input_ids)
    return embeds, inputs["attention_mask"].to(device)
