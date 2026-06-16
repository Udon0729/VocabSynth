"""既存トークナイザによる新語彙の分解。

新語彙文字列を、追加 **前** のトークナイザで分解し、
構成トークンID列とその文字列表現を返す。
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from vocabsynth.registry import VirtualToken, VocabularyRegistry


@dataclass
class TokenDecomposition:
    """トークナイザによる分解結果。

    Attributes:
        surface: 元の文字列。
        token_ids: 分解後のトークンID列。
        token_strings: 各トークンIDに対応する文字列。
    """

    surface: str
    token_ids: list[int]
    token_strings: list[str]


class TokenizerAnalyzer:
    """トークナイザによる新語彙の分解器。

    新語彙を追加する **前** のトークナイザを保持し、
    任意の文字列を既存語彙空間のトークン列に分解する。

    Args:
        tokenizer: 新語彙追加前のトークナイザ。

    Important:
        語彙追加後のトークナイザを渡すと、新語彙が単一トークンとして
        返されてしまい、合成材料が得られない。
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def decompose(self, text: str) -> TokenDecomposition:
        """文字列を既存トークン列に分解する。

        Args:
            text: 分解対象の文字列。

        Returns:
            トークンID列と対応する文字列表現を含む :class:`TokenDecomposition`。
        """
        encoded = self._tokenizer.encode(text, add_special_tokens=False)
        strings = [self._tokenizer.decode([tid]) for tid in encoded]
        return TokenDecomposition(
            surface=text,
            token_ids=encoded,
            token_strings=strings,
        )

    def analyze_registry(self, registry: VocabularyRegistry) -> dict[str, TokenDecomposition]:
        """レジストリ内の全仮想トークンを分解し、トークンID列を書き込む。

        各 :class:`~vocabsynth.registry.VirtualToken` の
        ``component_token_ids`` フィールドに結果を格納する。

        Args:
            registry: 仮想トークンのレジストリ。

        Returns:
            表層文字列をキーとする分解結果の辞書。
        """
        results: dict[str, TokenDecomposition] = {}
        for vtoken in registry:
            decomp = self.decompose(vtoken.surface)
            vtoken.component_token_ids = decomp.token_ids
            results[vtoken.surface] = decomp
        return results
