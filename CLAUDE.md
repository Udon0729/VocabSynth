# VocabSynth

Training-free Static Vocabulary Synthesis の実装・実験リポジトリ。

## 依存管理・実行

- パッケージ管理: `uv`
- 依存解決: `uv sync`
- スクリプト実行: `uv run python scripts/<script>.py`
- ドキュメント生成: `uv run sphinx-build docs docs/_build/html`

## GPU 使用規約

- 使用許可: GPU 4, 5, 6 のみ
- GPU 0-3 は使用禁止
- 実行時は `CUDA_VISIBLE_DEVICES=4` 等を指定

## モジュール構成

- `src/vocabsynth/registry.py` — VocabularyRegistry: 仮想トークンの定義と管理
- `src/vocabsynth/analyzer.py` — TokenizerAnalyzer: 既存トークナイザでの分解
- `src/vocabsynth/composer.py` — EmbeddingComposer: 構成トークンからの埋め込み合成
- `src/vocabsynth/injector.py` — VirtualInputInjector: 合成埋め込みの注入
- `src/vocabsynth/logit_head.py` — VirtualLogitHead: 出力側の仮想logit追加

## 実験

- `scripts/eval_input_equiv.py` — 入力同等性評価（第一段階）
- 結果出力先: `results/`
