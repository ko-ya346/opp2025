# opp2025

# 目標
- 何色でもいいのでメダル獲る。獲って会社と X で自慢する
- transformer の理論を理解して実装までやる
- 高分子ドメイン知識を存分に活かして知見共有や情報交換をする

# 実験・提出フロー
1. 実験毎にブランチを切る
2. 提出したい notebook を `submit_notebook/submit.ipynb` に置く
3.  


# アイディアメモ
- マルチタスク学習どうするか
- データ拡張
- データ生成
- 外部データセット利用
- データ表記法変更 SELFIES
- 官能基毎評価 → 剛直性、分子間結合、水素結合

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

