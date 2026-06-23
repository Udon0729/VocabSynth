"""仮想トークンの登録と管理。

新語彙の文字列、関係タイプ、構成要素、表示名を一元管理する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RelationType(Enum):
    """構成要素間の意味関係。"""

    NONE = "none"
    PLACE_FOOD = "place+food"
    PLACE_STRUCTURE = "place+structure"
    PLACE_INSTITUTION = "place+institution"
    MATERIAL_ARTIFACT = "material+artifact"
    PURPOSE_CONTAINER = "purpose+container"
    # 3語以上の合成向け
    PLACE_MATERIAL_ARTIFACT = "place+material+artifact"
    PLACE_FOOD_STYLE = "place+food+style"
    PLACE_STRUCTURE_FEATURE = "place+structure+feature"
    # 接辞による派生語向け
    PREFIX_DERIVED = "prefix+derived"
    SUFFIX_DERIVED = "suffix+derived"


@dataclass
class VirtualToken:
    """仮想トークンの定義。

    Attributes:
        surface: 新語彙の文字列表現（例: ``"HamamatsuGyoza"``）。
        components: 構成要素の文字列リスト（例: ``["Hamamatsu", "Gyoza"]``）。
        relation: 構成要素間の意味関係。
        display_name: 表示用の文字列。省略時は ``surface`` を使う。
        component_token_ids: :class:`TokenizerAnalyzer` が埋める構成トークンID列。
    """

    surface: str
    components: list[str]
    relation: RelationType = RelationType.NONE
    display_name: str = ""
    component_token_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.surface


class VocabularyRegistry:
    """複数の仮想トークンを管理するレジストリ。

    Examples:
        >>> reg = VocabularyRegistry()
        >>> reg.add("NaritaCake", ["Narita", "Cake"], RelationType.PLACE_FOOD)
        >>> reg["NaritaCake"].components
        ['Narita', 'Cake']
    """

    def __init__(self) -> None:
        self._tokens: dict[str, VirtualToken] = {}

    def add(
        self,
        surface: str,
        components: list[str],
        relation: RelationType = RelationType.NONE,
        display_name: str = "",
    ) -> VirtualToken:
        """仮想トークンを登録する。

        Args:
            surface: 新語彙の文字列。
            components: 構成要素の文字列リスト。
            relation: 構成要素間の意味関係。
            display_name: 表示用文字列。省略時は *surface* と同一。

        Returns:
            登録された :class:`VirtualToken`。
        """
        token = VirtualToken(
            surface=surface,
            components=components,
            relation=relation,
            display_name=display_name,
        )
        self._tokens[surface] = token
        return token

    def add_from_dicts(self, entries: list[dict]) -> None:
        """辞書のリストから一括登録する。

        Args:
            entries: 各辞書は ``surface``, ``components`` を必須キー、
                ``relation``, ``display_name`` を任意キーとして持つ。
        """
        for entry in entries:
            relation = entry.get("relation", "none")
            if isinstance(relation, str):
                relation = RelationType(relation)
            self.add(
                surface=entry["surface"],
                components=entry["components"],
                relation=relation,
                display_name=entry.get("display_name", ""),
            )

    def __getitem__(self, surface: str) -> VirtualToken:
        return self._tokens[surface]

    def __contains__(self, surface: str) -> bool:
        return surface in self._tokens

    def __iter__(self):
        return iter(self._tokens.values())

    def __len__(self) -> int:
        return len(self._tokens)

    def surfaces(self) -> list[str]:
        """登録済みの全表層文字列を返す。"""
        return list(self._tokens.keys())
