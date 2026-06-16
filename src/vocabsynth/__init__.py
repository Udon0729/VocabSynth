"""Training-free Static Vocabulary Synthesis.

既存のオープンウェイト自己回帰言語モデルに対して、追加学習なしで
新語彙を仮想的に追加し、入力側では単一トークン埋め込みとして、
出力側では通常語彙と同じlogit候補として扱えるようにする。
"""

__version__ = "0.1.0"
