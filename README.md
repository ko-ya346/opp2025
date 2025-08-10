# opp2025

# 目標
- 何色でもいいのでメダル獲る。獲って会社と X で自慢する
- transformer の理論を理解して実装までやる
- 高分子ドメイン知識を存分に活かして知見共有や情報交換をする

# 実験・提出フロー
1. 実験毎にブランチを切る
2. 提出したい notebook を `submit_notebook/submit.ipynb` に置く
3. PR 作成
4. github actions により`src` 配下のスクリプトがデータセット、`submit.ipynb` が kaggle notebook として PR 毎に生成される
5. web で notebook を開いて submit
6. スコア良好なら PR をマージ、悪かったら close （変更は破棄されるので src 配下を好き勝手いじれる）
---
> 同一 PR で notebook を修正した場合、web で notebook を削除する必要がある



# アイディアメモ
- マルチタスク学習どうするか
- データ拡張
- データ生成
- 外部データセット利用
- データ表記法変更 SELFIES
- 官能基毎評価 → 剛直性、分子間結合、水素結合

---
# 作業記録
## 6/19 9:00 ~ 10:00
- リポジトリ作成
- ディレクトリ構成決める
- ファイルダウンロード
- uv で仮想環境作成
- jupyter lab 環境設定
- ベースライン用の参考コードを眺める
  - https://www.kaggle.com/code/senkin13/baseline-rdkit-descriptors-features

### next action
- データ眺める
- 学習パイプラインを実装して試しに学習してみる

### メモ
- データサイズが小さいのである程度までならローカルで学習できそう
- uv 爆速で快適

## 6/21 5:00 ~ 6:00
- lgb パイプライン作成 exp001 提出 -> LB 0.1 くらい
- データ拡張、特徴量追加を試す
- rdkit 素振り

## 6/21 21:30 ~ 22:30
- CV 計算の実装について調査
  - https://www.kaggle.com/code/richolson/smiles-rdkit-lgbm-ftw?scriptVersionId=246623061&cellId=25
  - ロジックが謎なので引き続きやる

### next action
- 外部データ使う
  - https://www.kaggle.com/datasets/minatoyukinaxlisa/smiles-tg/data

## 6/22 15:30 ~ 16:20, 24:30 ~ 25:30
- Tg, Tc の外部データ追加 -> exp002
- CV 計算ロジック追加
- 前処理追加
- morganfingerprint が学習に使われていなかったので修正
- Code 調査
  - https://www.kaggle.com/code/abdulrahmanqaten/neurips-third-solve
    - xgb, lgb, deberta アンサンブル で 0.059
    - 外部データ未使用
    - 目的変数ごとにアンサンブル
  - https://www.kaggle.com/code/richolson/smiles-rdkit-lgbm-ftw
    - 外部データ使って lgb, 外部データは concat のみ
    - 損失関数が mae

## 6/22 8:50 ~ 10:00
- exp002 をちょっと調整
  - 特徴量を正規化
  - ハイパーパラメータをちょっと調整
- torch, transformers をインストールして動作確認

### next action
- transformers 使う
- マルチタスク学習を考える
- 学習データを眺めてクレンジングとかアイディア練る

## 6/23 22:00 ~ 23:30
- transformers の実装

## 6/24 5:40 ~ 6:30
- transformers の実装を理解する

## 6/24 8:40 ~ 10:00
- exp004, transformers で KFold とスコア計算を実装、なんかイマイチ

## 6/25 8:30 ~ 10:00
- exp005 transformers でスコア向上検討
  - 目的変数の正規化 -> スコア向上
  - optimizer, scheduler 設定
  - 損失関数調整
  - モデル変更
  - augumentation
  - マルチタスク学習
  - 今のところ良さがあんまり出てない

### next action
- transformers で拡張データ使う
- 外部データセット調査、データ生成
- 問題設定再調査

## 6/26 0:30 ~ 
- この人の discussion が示唆に富んでいる https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/585022
  - 類似コンペ https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/585022#3226316
  - GNN https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/585022#3226363
  - 公開データセット  https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/585022#3226388
  - 関連論文 https://arxiv.org/abs/2503.23491
  - Tg 予測には GNN が効果的らしい
- データを眺める
  - Ge は [Ge], Si は [Si] と入ってる. transformer でまとめて変換したほうがよさそう
  - スラッシュ, ハイフンが含まれてる。理由は謎

## 6/26 8:50 ~ 9:45
- 繰り返し単位 2 回の smiles を学習データに追加 -> 効果なし
- tokenizer の辞書確認 -> 文字単位ではなく部分文字単位になっていた、なので [Ge] とかは考慮してもらってる
- Tg, Tc の拡張データ追加

### next action
- GNN 実装

## 6/27 8:50 ~ 10:10
- GNN 実装 exp006
- https://www.kaggle.com/code/abdulrahmanqaten/neurips-gnns-solve-deep-learning
- Tg だけ推論してみたが、transformer よりも MAE 少ないのでよさげ

## 6/27 22:10 ~ 23:10
- GNN 学習 exp006
- 眠すぎてダメ

### next action
- target 正規化
- oofデータ格納方法変更
- lightgbm で morgan fp の次元増やす
- lightgbm で oof 予測を特徴量にして別ターゲット学習

## 6/28 13:10 ~ 14:30
- GNN の CV 計算部分修正 -> 0.08

## 6/30 9:00 ~ 10:10
- exp009 で実装を分割し始めた
  - src.data 配下で関数オブジェクトとして定義（各ファイルで 1 処理を定義して __init__.py で読み込み）
  - `from src.data import hogehoge` みたいに呼び出す

## 7/1 8:30 ~ 10:00
- LB みたら 0.037 台がめちゃ増えてた
  - Code で高スコアの実装が公開されてた、外部データマシマシ
  - 参考にしたがスコア良くならず
  - 元データを入れなおしたり特徴量を絞ったりして提出してみた（Version 12）
  - CV は 0.053、LB は 0.121 （Version 10） で外部データの質が問われている
  - 提出時に外部データ使うとスコアが良化するので、学習時には使わないほうがいいかもしれない
  - Version 12 が 0.067 でスコア上がらない理由が謎
  - 特徴量スケーリングが悪さしてるかもしれないので除外（lightgbm だと不要） -> version 13
- 
- CV の計算方法、solution の null 埋め有無で 0.006 変わる（埋めないほうがよい）。理由はよくわからない
- コードを分割中。exp010 で GNN モデルの分割完了。まだ提出用コードの準備はできていない
- github リポジトリの容量オーバーで変になったのでリポジトリ作り直した
- 学習データについて色々検討されてそうなので注力する

### next action
- X でコンペの情報収集
- Smiles の特徴量を使って NN
- GNN モデルの埋め込みと Smiles の特徴量を使って NN
- GNN モデルの埋め込みを LightGBM の特徴量に使ったほうがよいかもしれないので検討する
- Discussion よむ

## 7/2 5:30 ~ 9:00
kaggle dataset をアップロード  

1. kaggle の設定から API 取得（username, secret_key）
2. upload したい階層に `dataset-metadata.json` を用意
3. コマンド実行  

```
# フォルダをアップロードする場合
kaggle datasets create -p test_upload_dir -- dir-mode zip
# ファイルをアップロードする場合
kaggle datasets create -p test_upload_dir
```

## 7/3 0:00 ~ 1:40
- 提出用の github actions 実装

### next action
- 実際に提出 -> submit までの流れを行ってみる
- 提出フローのドキュメント生成（chatGPT に投げる）
- CV - LB の相関確認
- 外部データ使わずに実験（外部データ使用できなくなる可能性があるので、モデリングで差別化できるようにする）

## 7/3 8:30 ~ 9:30
- 提出用notebook を作成中 -> `submit_notebook/submit.ipynb`
- src の import 周りで苦戦中

## 7/4 8:30 ~ 9:00
- 提出フローだいたい完成
  - src/ をそのままデータセット化できなかった。その下の階層が展開されるようだ
  - kaggle kernels の設定 json で存在しないデータセットを指定した場合、エラーにならず無視される
  - kaggle notebook で import したデータセットは read only で編集不可

### next action
- 外部データセット使わない場合のスコアを lightgbm, GNN, transformer それぞれ再確認
- W&B でスコア管理


## 7/12
- exp014 で GNN 提出
- CV と LB の乖離が大きい、kfold の分割が良くない可能性があるので scaffold で分子骨格を取り出して GroupKFold した exp015, pr005

### next action
- ruff で lint, format
- groupkfold の各グループ傾向把握
- GREA 調査
- SGIR 調査 (半教師あり学習)
- GNN の特徴量追加
- 繰り返し単位増やしてデータ増やす
- GNN に分子記述子や指紋の情報を追加

## 7/13
- groupkfold しても CV と LB の乖離解消せず
- Tg のエラーの傾向確認... SMILES長さや原子の種類とエラーに明確な相関なし
- スコア向上のための手法を調査
  - 原子に対する情報を増やす
  - 結合に対する情報を増やす
  - global pooling の工夫
  - フィンガープリントなどの特徴量を混ぜる
  - GREA とか他のモデルを試す
- `uv add torch-molecule` 実行したところ python 3.10 以上が必要だった
- 以下コマンドでインストールと固定化実施

```
uv python install 3.10
uv python pin 3.10
uv add torch-molecule
```

- exp016 GNN のモデルを torch-molecule に置き換え。学習コード修正中

### next action
- 学習と推論を分離
- 推論 notebook 作成
- github actions でモデルパラメータをデータセットに上げる
- パラメータ探索用スクリプトを作って寝ている間に探索させる
- torch-molecule の色々なモデルを試す

## 7/14 8:40 ~ 9:25
- テストデータ推論してなかったので修正
- exp016, CV: 0.160, LB: 0.258 まだ乖離してる

## 7/15 8:40 ~ 9:30
- torch-molecule のデータセット用意した
- 外部データセット使わずに提出 → スコア下がった
- torch-molecule の ソースコード読んで使っている特徴量やパラメータをある程度理解した
  - morganfingerprint あとで追加しようと思っていたがすでに使われていた
- exp017 で epochs を 500 に増やした

## 7/16 8:40 ~ 9:30
- 学習済モデルを kaggle dataset にアップロードする関数実装
  - kaggle notebook への import は手動でやります
- submit.ipynb を学習済モデル使うように変更
- get_model 関数作った
- SGIR モデルはマルチタスクできない
- アンサンブルを chatgpt さんに相談

## 7/16 22:20 ~ 23:30
- exp018 で gnn, grea, exp019 で sgir を学習できるように整理
- notebook 実行スクリプト作成 `scripts/run_notebook.sh`

### next action
- アンサンブル検証
- lightgbm のコード整理
- catboost 実装
- テーブルデータ向け NN 試す
  - https://zenn.dev/mkj/articles/f7939cb221da14
- epoch 数をスクリプト側で上書きできるようにする（papermill -> run）

## 7/17 
- ポリマー末端の * の扱い
  - C に置き換え、未処理、At で置き換え
  - タスクによって精度が異なる
  - 3D モデルで異なる可能性（3D モデルとは？）
  - https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/589372
- シミュレーションによるデータ作成はけっこう時間かかる
  - それで精度上げることはこのコンペの目的に合ってない


```
新しい Kaggler は、なぜデータ漏洩について知っている人がいるのかといつも不思議に思っています。
それは、彼らがデータをより深く分析しているからです。
新しい Kaggler はモデリングに重点を置く傾向がありますが、古い Kaggler はそれが DATAsience であって
MODELscience ではないことを知っています。
```

- 繰り返し単位を追加することは有効
  - データ拡張ではなく検証サンプルや追加トレーニング用につかう
  - 繰り返し単位で増やして拡張し、元の SMILES でグルーピングして最もスコアが高いものを採用する（もしくは平均、中央値など）
  - https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/589360#3248754
- GNN で descriptor 特徴量増やしたい
  - augument を自作
  - torch_molecule を pyproject.toml からはずして リポジトリを src 配下に
  - import エラー発生

### next action
- notebook エラー回避
- GNN 特徴量拡張
- SMILES 正規化
- アンサンブル提出 exp021

## 7/21 
- exp021 アンサンブルのスコア確認 -> 悪くなった
- exp022 torch_molecule を改造して Descriptors の結果を特徴量に追加
- SMILES を canonical にして exp018, exp019, exp020 学習しなおし

### next action
- atom_to_feature_vector に特徴量追加（ハロゲンの index 追加したい）
- ほかに特徴量追加できないか考える
- W&B で実験結果管理（score, ターゲット毎の MSE）
- グラフ特徴量追加
- PolyBERT の埋め込み特徴量追加
  - https://www.kaggle.com/code/akihiroorita/rdkit-w-polybert-extra-neurips-lb-0-039#Final-Model-For-Submission
  - PolyBERT ってなに？
- networkx 使った特徴量

## 7/22
- GNN に descriptor の特徴入れたものはスコア上がった
- wandb 導入、fold 毎の loss は table に押し込んだ

### next action
- bert の埋め込みを追加

## 7/23
- exp022 でスコアが悪かったので、Descriptors の特徴量を選別
  - ユニーク数が 1つ、相関係数が 0.9 以上のものを除外 -> exp023
- networkx の特徴量を生成する関数実装
- 原子をカウントする関数実装
- bert で埋め込み生成し、PCA で次元圧縮する実装を確認

## 7/25
- exp018, exp020 のoof 予測値と正解の分布を確認、どちらのモデルもそこそこ予測できている
- そもそも目的変数に外れ値が含まれており、それらが悪さしている可能性があるので確認中
- Tg だとイミド環を含む化合物、多環（ナフタレンのような）の化合物が Tg かなり高く、モデルでとらえきれていない
  - イミド系は lgb が得意、多環化合物は GNN が得意そう
  - descriptors や morganfingerprint の特徴でどれくらい抑えられているか確認する

## 7/29
- exp023 のもろもろの実験結果を確認
  - desc 入れると悪くなる。ちっ
  - maccs も入れると悪い
  - 細かいところはあとでチューニングしよう
- exp024 graph_features, count_atoms の特徴量追加
  - exp020 と比べて CV 0.0005 改善
 
## 7/30
- exp024 を提出したところ LB が 0.287 と大きく乖離している。なぜだ
  - 外部データの有無では変わらない（version 4 が外部データなし、version 5 が外部データあり）
  - check_submit.ipynb で学習データに対して推論掛けたらスコア 0.03 くらいなので、推論の処理自体に問題はなさそう
  - 過学習してる可能性が高い
    - 不要な特徴量を排除する
    - 過学習しない学習条件を作る
- どうやら テストデータの Tg の単位が揺れているらしい
  - https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/591394
  - 提出時に Tg をケルビンにしたらスコア良化した。K と oC が混じっている？らしい
  - 虚無感が増してきた、、、

## 7/30
- Code を見たり、思い付きで実験することが不毛な感じがしてきたので一旦やめた
- 予測結果と正解を見比べて、精度向上のヒントを探し始めた
- check_oof.ipynb で誤差が大きい化合物を取り出して可視化できるようにした
  - これだけでは、特徴量が不足しているのか類似サンプルが少ないのか不明
- check_tsne.ipynb で特徴量を低次元に落として、誤差が大きいものとそうでないものの散布図を作成
  - 傾向あるようなないような。。。？
- この作業は考えること多くて結構楽しい。ドメイン知識も活かせそう

### next action
- tsne の値を使って、誤差が大きいサンプルと似ているサンプルを取り出す

## 8/1
- exp025 で トリマーを繋げた環状化合物の smiles を生成して gnn に渡してみた
  - Tc, Tg の熱物性においては精度向上が見られた
  - ほかの物性ではあまり変わらず
  - よく見ると、mf の特徴量の有無も目的変数によって効いたり効かなかったりしている
  - 目的変数毎にチューニングが必要そうだ

## 8/2
- exp025 の結果が良好だったので考察
  - Tg, Tc は分子の剛性、結合パターンが強く影響する。環状化により分子末端の自由度が消失し、環内ひずみが加わる情報を GNN でとらえやすくなり、予測精度が改善
  - FFV (Fractional Free Volume), Rg (分子の大きさ指標), Density は主に分子全体の体積、形状、分子間配向挙動に依存
  - GNN の局所的メッセージパッシングだけでは全体ボリューム感、分子間パッキングを十分にとらえきれない可能性がある
  - 改善案
    - 3D 構造特徴量の導入
      - mordred (コミュニティによる互換バージョンがある、2D 記述子は rdkit よりも多彩)
      - 3D 対応GNN (SchNet, DimeNet, PaiNN), 原子の座標と種類をそのままネットワークに入力し、メッセージパッシングとともに距離情報を学習
      - 結合情報だけでなく原子間距離・角度・三体相関などエッジ特徴として与えることで立体構造依存性の高い性質予測に強い
    - Global Readout 層の工夫 （Attension 機構、Set2Set のようなグローバルプーリングで分布館をとらえる）
- mordred の記述子を計算する関数実装、exp026 で精度検証
  - numThreads = 10 で 8600 行データの処理に 1-2 時間かかりそう
  - mordred の特徴量だけだと良化しないので、rdkit の記述子と併用してみる

## 8/3
- MolMix 触る
  - https://github.com/andreimano/MolMix
  - 1D, 2D, 3D のモデル全部使える
  - nvcc が必要そうなので準備

## 8/8
- MolMix を試そうとして Ubuntu アップデートしたら依存関係こわれた
  - 復旧するのに 1週間弱かかってしまった
- その間に LB が正常化していた
  - 0.074, gnn と lightgbm いずれも
  - gnn は CV もっと悪かった気がするけど、、、

### next action
- スコアいい Code が出ていたので参考にしてみる
  - 追加データが要因ぽく見えるけど、、、
  - データ拡張とか色々やってるけど意味あるようにはみえない
  - https://www.kaggle.com/code/guanyuzhen/neurlps-2025-baseline-random?scriptVersionId=250870055
- MolMix を google cloud で触ってみる
  - flash_attn を書き換える
 
## 8/9
- chatgpt-5 に壁打ちしたら自分の知識を超えたアドバイスをたくさんもらった
- ターゲット同士の相関確認... check_corr_target.ipynb
  - あんまり相関なし、この場合はマルチタスク学習はイマイチらしい
- target 毎の平均 mae を wandb に記録できるようにした (exp018, exp025, exp026)
- run_notebook.sh に実験セット
  - scaffold で groupkfold
  - gnn で シングルタスク学習 