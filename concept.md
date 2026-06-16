以下では，提案手法を Training-free Static Vocabulary Synthesis と仮に呼ぶ。目的は，既存のオープンウェイトな自己回帰Transformer LMに対して，追加学習なしで新語彙を仮想的に追加し，入力側では1 token embeddingとして扱い，出力側では通常語彙と同じlogit候補として扱えるようにすることである。

この設計は，既存研究でいう vocabulary expansion や tokenizer transfer と近いが，それらの多くが「新token embeddingの初期化後に継続事前学習またはfine-tuningする」ことを前提にするのに対して，本設計では 合成したembedding / output weightをそのまま使う。したがって，研究上の焦点は「よい初期化」ではなく，「既存語彙空間上の明示的合成だけで，どこまで通常tokenとして振る舞えるか」である。

1. 対象モデルとアクセスすべき機構

対象は，Hugging Face Transformersで扱える causal LM とする。最初は構造が単純で，embeddingとlm_headに直接アクセスしやすいモデルを使うのがよい。候補は GPT-2 系，Pythia 系，Qwen 0.5B〜1.5B級などである。Hugging FaceのPreTrainedModelには入力embeddingを取得・変更するための共通メソッドがあり，resize_token_embeddingsは語彙数変更時に入力embedding行列をリサイズし，モデルが対応していれば重み共有も処理する，と公式ドキュメントで説明されている。 

アクセスすべき機構は四つである。

第一に，tokenizerである。新語彙 z を追加し，既存語彙IDとは異なる新しいIDを割り当てる。ただし，合成に使う構成token列を得るときには，新語彙を追加する前の tokenizer で z を分解する必要がある。たとえば HamamatsuGyoza を新tokenとして追加する場合，追加前tokenizerで Hamamatsu, G, yo, za などに分解される。その分解結果が合成材料になる。

第二に，入力embedding行列 E ∈ R^{V×d} である。ここで V は既存語彙数，d は隠れ次元である。model.get_input_embeddings().weight からアクセスする。新語彙 z には新しい行 E_z ∈ R^d を与える。これはランダム初期化ではなく，既存token embeddingの合成で作る。

第三に，出力head，すなわち lm_head の重み W ∈ R^{V×d} である。自己回帰LMでは，最後のhidden state h_t ∈ R^d に対して logits = h_t W^T + b を計算する。新語彙を出力候補にするには，W にも新語彙用の行 W_z を追加する必要がある。入力embeddingとlm_headがweight tyingされているモデルでは，E_z と W_z を同じベクトルとして扱える可能性がある。weight tyingされていないモデルでは，入力側と出力側を別々に合成する。

第四に，forward時の inputs_embeds 経路である。実装初期段階では，tokenizerやembedding層を本当にリサイズせず，input_ids から一度embeddingを取り出し，新語彙位置だけを合成embeddingで置き換えて inputs_embeds としてモデルに渡す方が安全である。これにより，モデル本体の構造変更を避けたまま，合成embeddingの挙動を検証できる。出力側の評価に進む段階で，lm_headに仮想logitを追加する。

2. 全体アーキテクチャ

設計は，三つの層に分ける。

一つ目は Lexical Composer である。これは，新語彙文字列 z を既存token列 c(z) = [t_1, …, t_k] に分解し，既存embeddingから E_z を作る部分である。

二つ目は Relation Corrector である。これは，単純平均では表現しにくい複合語関係を，既存語彙空間または既存phrase表現から補正する部分である。たとえば「地名＋食品 → ご当地食品」のような関係方向を推定し，E_z に加える。

三つ目は Virtual LM Head Extension である。これは，通常語彙のlogitに加えて，新語彙 z のlogitを計算する部分である。通常の lm_head は V 個のlogitを出すが，本設計では追加で m 個の仮想新語彙logitを計算し，V+m 個の候補として扱う。

流れは次のようになる。

raw text
→ tokenizer with virtual tokens
→ new token位置を検出
→ Lexical Composerで E_z を生成
→ embedding列に E_z を挿入
→ frozen Transformer blocks
→ hidden state
→ 通常 lm_head による既存語彙logits
→ Virtual LM Head Extensionによる新語彙logits
→ 既存語彙logitsと新語彙logitsを連結

この設計では，Transformer block，attention，MLP，layer norm，position embedding / RoPEは変更しない。変更するのは，入力embeddingの生成方法と，出力logit候補の追加だけである。

3. 新語彙embeddingの合成

最小構成では，新語彙 z の構成token列を t_1, …, t_k とし，入力embeddingを

E_z = normalize(Σ_i α_i E_{t_i})

で作る。normalize は，合成ベクトルのノルムを既存embedding分布に合わせる処理である。たとえば，構成tokenの平均ノルムに合わせるか，既存語彙embedding全体の平均ノルムに合わせる。

重み α_i には三つの基準を用意する。

最初のベースラインは単純平均である。

α_i = 1/k

次に，後部主要部を重くする合成を使う。日本語や英語の複合名詞では，後部要素がカテゴリを決める場合が多い。たとえば Hamamatsu Gyoza では Gyoza が食品カテゴリを決める。そのため，

E_z = 0.4 E_place + 0.6 E_food

のようにする。

三つ目は，文字列長またはtoken長で重み付けする方法である。BPEやSentencePieceでは，短い断片tokenが意味単位とは限らないため，より長いtokenを重くする。たとえば token_length(t_i) を使って，

α_i = len(t_i) / Σ_j len(t_j)

とする。

この段階は，既存研究でいう embedding initialization に近い。WECHSELは，既存LMを新言語へ効率よく移すためのsubword embedding初期化手法として提案されており，語彙拡張時のembedding配置が重要であることを示す代表的研究である。  2024年のMundraらの研究も，RoBERTaとLLaMA 2を対象に語彙拡張時の初期化手法を比較し，既存embeddingの凸包内に新embeddingを置くことの理論的・実験的妥当性を扱っている。  ただし，これらは多くの場合，初期化後の継続学習を前提とする。本設計は，その初期化を最終表現として使う点が異なる。

4. 関係補正ベクトルの設計

単純平均だけでは，浜松 と 餃子 の混合にはなるが，「浜松に由来する餃子」「浜松のご当地料理」という関係が弱い。そこで，複合語関係を既存語彙空間から抽出する。

関係タイプ r をあらかじめ限定する。最初は，place+food，place+institution，material+artifact，purpose+container のような少数に絞る。たとえば place+food では，参照ペア集合 P_r を用意する。

P_place_food = { (Utsunomiya, Gyoza, UtsunomiyaGyoza), (Hakata, Ramen, HakataRamen), ... }

各参照ペアについて，構成要素から作ったベース合成と，複合語表現との差分を取る。

δ_i = E(compound_i) - compose_base(E(head_i), E(modifier_i))

そして，関係補正ベクトルを

R_r = average_i δ_i

とする。新語彙 z には，

E_z = normalize(compose_base(z) + λ R_r)

を使う。λ は学習しない固定係数である。最初は λ ∈ {0, 0.25, 0.5, 1.0} のようにグリッドで比較する。これは追加学習ではなく，設計上の非学習的ハイパーパラメータである。

問題は，UtsunomiyaGyoza のような複合語が単一tokenとして存在しないことが多い点である。この場合，三つの実装選択肢がある。

第一は，単一tokenとして存在する複合語だけを使う方法である。これは最も純粋な静的合成型である。ただし，日本語では候補が少ない。

第二は，複合語phraseを既存token列としてモデルに入力し，最後のtoken位置または平均poolingでphrase表現を得る方法である。これは内部表現を使うため，厳密な静的embedding合成からは少し外れる。しかし，関係補正の推定だけに使い，最終的な新語彙embeddingは静的に固定するなら，実装上は現実的である。

第三は，外部辞書や説明文を使わず，既存embeddingの近傍から類似複合語候補を作る方法である。たとえば place と food の候補集合を用意し，既存語彙に含まれる結合形だけを探索する。これはオープンウェイトモデルだけで完結しやすい。

このRelation Correctorは，ZeTTやHyperOFAとは異なる。ZeTTは，任意の新tokenizerに対してtoken embeddingを得るため，tokenizerを入力に取るhypernetworkを訓練する研究である。  HyperOFAも，新token embeddingを生成するhypernetworkを訓練し，OFAのような凸結合ベース初期化の表現力制約を補おうとする研究である。  本設計では，hypernetworkを訓練せず，既存embedding空間上の明示的演算だけで関係補正を作る。

5. 入力側実装

入力側には二つの実装段階を置く。

第一段階では，tokenizerを本当に拡張しない。テキスト中の新語彙候補を前処理で検出し，そのspanを特殊プレースホルダに置き換える。通常tokenizerで前後文脈をtokenizeし，プレースホルダ位置に E_z を直接挿入する。モデルには input_ids ではなく inputs_embeds を渡す。この方法では，語彙表やembedding層のサイズを変更しないため，実験が壊れにくい。

第二段階では，tokenizerに新tokenを追加し，resize_token_embeddings でembedding行列を拡張する。追加された行に，合成済み E_z を書き込む。出力側まで含める場合は，lm_head 側にも対応する行を追加する。Hugging Faceでは，tokenizerが入力前処理を担い，モデル側は入力embeddingのresizeやweight tyingを扱うため，この段階は既存APIと整合する。 

設計書としては，第一段階を研究検証用，第二段階を実運用に近い実装用と位置づけるべきである。最初から第二段階に進むと，tokenizer追加，special token扱い，embedding resize，lm_head tying，保存・読み込みの問題が絡み，合成規則の良し悪しを切り分けにくい。

6. 出力側実装

出力側では，最後のhidden state h_t に対して，新語彙logitを明示的に追加する。

通常語彙logitsは，

l_vocab = h_t W^T

である。新語彙集合を Z = {z_1, …, z_m} とし，それぞれに出力weight U_z を作る。weight tyingモデルなら U_z = E_z とする。weight tyingされていないモデルなら，出力head重み W の構成token行から別途合成する。

U_z = normalize(Σ_i α_i W_{t_i} + λ R^W_r)

そして，

l_virtual = h_t U_Z^T

を計算し，

l_all = concat(l_vocab, l_virtual)

とする。これにより，新語彙は通常語彙と同じsoftmax候補に入る。

ここで必須なのは，スケール補正である。U_z のノルムが既存 W の分布から外れると，新語彙logitが過小または過大になる。そのため，U_z のノルムを，構成tokenの出力weightノルム平均，または既存語彙全体の平均ノルムに合わせる。さらに，必要なら非学習的なlogit biasを導入する。ただし，biasを導入すると生成頻度を人為的に動かすため，研究主張では「biasなし」と「norm補正のみ」を主条件にする方がよい。

この出力側拡張は，dynamic vocabulary研究とは関係するが，同一ではない。Generation with Dynamic Vocabularyは，生成時に任意のtext spanを基本生成単位として扱い，multi-tokenを原子的に生成することで品質や効率を改善する研究である。  その研究はspanを生成単位として扱う方向であり，本設計のように「既存embedding / lm_head weightから新しい単一token weightを明示合成する」ことを主目的にしているわけではない。

7. 推奨する最小実装

最小実装では，モデルは gpt2 または EleutherAI/pythia-410m 程度にする。日本語例を最初から中心に置くと，tokenizer品質，事前知識，表記揺れが絡むため，初期検証では英語またはローマ字複合語を使う方がよい。日本語は，設計が動くことを確認した後にQwen系や日本語対応モデルで試す。

実装モジュールは以下の五つに分ける。

VocabularyRegistry は，新語彙文字列，関係タイプ，構成token列，表示文字列を保持する。たとえば HamamatsuGyoza に対して，構成要素を Hamamatsu と Gyoza，関係タイプを place+food として持つ。

TokenizerAnalyzer は，新語彙を追加する前の tokenizer で分解し，構成token ID列を返す。ここでは，語彙追加後tokenizerを使ってはいけない。追加後tokenizerを使うと，新語彙そのものが1 tokenとして返ってしまい，合成材料が失われる。

EmbeddingComposer は，入力embedding行列 E と構成token ID列から E_z を作る。単純平均，後部主要部重み，長さ重み，関係補正あり，の各方式を実装する。

VirtualInputInjector は，通常embedding列のうち新語彙spanに対応する位置を E_z で置き換え，inputs_embeds とattention maskを作る。

VirtualLogitHead は，通常のlm_head出力に対して，新語彙logitを追加する。出力評価では，通常語彙IDと仮想語彙IDの対応表を持ち，top-kに新語彙が入ったとき表示文字列へ戻す。

この分け方にしておくと，後から本当にtokenizerを拡張する実装へ移行しやすい。

8. 評価設計

評価は三段階に分ける。

第一段階は 入力同等性評価 である。これは，新語彙を複数token列として入れた場合と，合成1 tokenとして入れた場合の挙動を比較する。プロンプト X is a local food from ___ のような文を用意し，X を複数token版と合成token版で入れ替える。比較指標は，次token分布のKL divergence，top-k overlap，指定候補語の順位，最後のhidden stateのcosine similarityである。

ここで合成token版が複数token版に近ければ，入力圧縮としては機能している。近くなければ，出力生成評価へ進む前に合成規則を見直す必要がある。

第二段階は 意味推論評価 である。これは，合成tokenが構成要素間の関係を反映しているかを見る。たとえば HamamatsuGyoza is a famous local ___ に対して，food, dish, specialty の確率や順位を見る。place+food補正ありの方が，単純平均よりもカテゴリ候補を上げるかを見る。

ここでは，実在語と人工語を分けるべきである。実在語では，モデルが事前学習で既に知っている可能性がある。人工語では，文脈と構成要素だけから推論できるかを見る。たとえば NaritaCake，OsakaNoodle のような人工複合語を作り，説明文脈を与える。

第三段階は 出力候補評価 である。文脈 A famous local dumpling from Hamamatsu is called ___ のような入力に対して，仮想新語彙 HamamatsuGyoza のlogit順位を見る。この段階で初めて，入力側だけでなく出力側の合成weightの妥当性を評価する。

比較条件は，少なくとも次を置く。

multi-token baseline は，新語彙を追加せず既存token列のまま扱う条件である。

random virtual token は，新語彙embedding / output weightをランダムにする条件である。

mean composition は，構成tokenの単純平均である。

head-weighted composition は，複合語の主要部を重くする条件である。

relation-corrected composition は，関係補正ベクトルを足す条件である。

研究上の最低限の成功条件は，random virtual token より合成型が明確に良く，mean composition より relation-corrected composition が意味推論評価で良いことである。出力候補評価は難しいため，初期段階では入力同等性と意味推論を主評価にし，出力候補評価は副評価として扱う方が堅い。

9. 先行研究との位置づけ

本設計に最も近い周辺研究は四系統に分かれる。

第一は，語彙拡張時のembedding初期化である。WECHSELは，既存LMを新言語へ転移するため，subword embeddingを効果的に初期化する手法である。  Mundraらの2024年研究は，RoBERTaとLLaMA 2に対して複数言語・複数タスクで語彙拡張と初期化手法を比較し，既存embeddingの凸包内に置く初期化の妥当性を議論している。  これらは本設計の「既存embeddingから新embeddingを作る」という部分に対応する。ただし，通常は継続学習の初期値として使うため，本設計のtraining-free運用とは目的が異なる。

第二は，tokenizer transferである。ZeTTは，学習済みLMを任意の新tokenizerに移す問題を定義し，新tokenizer語彙のembeddingを得るためにhypernetworkを訓練する。  本設計も「既存LMをtokenizer制約から部分的に解放する」という点では近い。しかし，本設計はhypernetworkを使わず，既存語彙空間上の明示演算だけで新語彙を作る。

第三は，inner lexicon / intrinsic detokenizationである。From Tokens to Wordsは，LLMがsubword列を早期〜中間層で単語表現へ統合することを示し，out-of-vocabulary wordsに対しても内部表現を入力ベクトルとして使うことで理解できる可能性を示している。また，fine-tuningなしの語彙拡張応用も述べている。  これはtraining-free語彙拡張という点で最も近い。ただし，本設計は中間層表現を抽出して使うのではなく，既存embedding / lm_head空間上の明示的な静的合成を使う点で異なる。

第四は，dynamic vocabulary / span generationである。Generation with Dynamic Vocabularyは，任意のtext spanを生成時の基本単位として扱うことで，multi-tokenを原子的に生成するplug-and-playな方法を提案している。  本設計はspanを生成単位として扱うだけではなく，新語彙に対応する入力embeddingと出力weightを明示的に構成する。そのため，デコード拡張ではなく，静的合成型の語彙拡張である。

補助的な関連として，VOLTは語彙選択そのものを最適輸送問題として捉え，trial trainingなしで良いtoken dictionaryを探す研究である。  これは新語彙embeddingを合成する研究ではないが，「語彙は単なる前処理ではなく，モデル性能と効率に関わる設計対象である」という位置づけを支える。

10. 本設計の新規性と限界

本設計の新規性は，「新語彙embeddingの初期化」ではなく，入力embeddingと出力lm_head weightの両方を，既存語彙空間上の明示的合成だけで構成し，追加学習なしで通常tokenとして使う点にある。さらに，単純平均ではなく，既存語彙空間から抽出した複合語関係方向を加えることで，place+food のような構成要素間関係を反映させる。

限界は明確である。第一に，この方法は新しい事実知識を獲得しない。浜松餃子 について，構成要素から「浜松に関係する餃子」までは推測できても，具体的な調理特徴や歴史的事実は，モデルが事前に知っているか，文脈で与えられない限り出てこない。

第二に，出力側は入力側より難しい。入力側では，合成embeddingが文脈内でそれなりに処理されればよい。しかし出力側では，logitスケール，ノルム，softmax内での競合，既存token列との競争が問題になる。したがって，出力生成で強い結果を出すには，norm補正と比較設計が必須である。

第三に，関係補正ベクトルの品質は，参照ペア集合に依存する。参照ペアが少ない，あるいはtokenizer上で複合語が単一tokenとして存在しない場合，純粋な静的合成だけでは補正が不安定になる。その場合，中間層phrase表現を関係推定にだけ使う折衷案が必要になるが，その場合は設計の純粋性がやや下がる。

11. 実装順序

最初に行うべきなのは，inputs_embeds を使った入力側評価である。tokenizerを拡張せず，新語彙spanを合成embeddingに置き換え，複数token baselineとの次token分布差を見る。ここで挙動が近くならない場合，語彙追加やlm_head拡張に進む意味は薄い。

次に，tokenizerを実際に拡張し，追加embedding行に合成ベクトルを書き込む。ここでは，保存・読み込み，special token扱い，paddingやattention maskの挙動を確認する。

その後，lm_headに仮想logitを追加する。最初はモデル本体のlm_headを物理的に拡張せず，forward後に h_t · U_z を別計算して通常logitsに連結する。これにより，既存モデルの重みファイルを壊さずに出力評価できる。

最後に，Relation Correctorを追加する。最初は place+food のような一種類だけでよい。単純平均，主要部重み，関係補正ありを比較し，意味推論タスクで改善するかを見る。

この順序なら，ゼロから訓練せず，既存オープンウェイトモデルの推論だけでプロトタイプを作れる。研究としての最小主張は，「既存語彙embeddingとlm_head weightの明示的合成だけで，新語彙を入力・出力の両方に追加できるかを，training-free条件で評価する」である。既存研究と重なる部分はあるが，合成weightを初期値ではなく完成表現として使い，さらに出力側まで同時に扱う点を明確にすれば，設計上の独立性は保てる。
