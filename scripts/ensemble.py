from pathlib import Path

import os
import sys
import json
import argparse

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sqlalchemy.engine import create

abspath = Path("/home/kouya-takahashi/kaggle/opp2025/")
sys.path.append(str(abspath))

from src.data import load_data
from src.utils import NULL_FOR_SUBMISSION, score
from src.utils.upload_kaggle_dataset import create_kaggle_dataset_metadata, upload_kaggle_dataset

# --- 追加: モデル選択ユーティリティ -----------------------------------------
from collections import OrderedDict
from sklearn.linear_model import Ridge

def per_model_mae_table(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """
    各モデル列ごとに、(y, そのモデル列) がともに非NaNな行で MAE を計算
    """
    maes = OrderedDict()
    for col in X.columns:
        m = (~np.isnan(y)) & (~X[col].isna().values)
        if m.sum() == 0:
            maes[col] = np.inf
        else:
            maes[col] = mean_absolute_error(y[m], X.loc[m, col].values)
    return pd.Series(maes).sort_values()

def select_models_by_mae(mae_s: pd.Series, rel_delta: float = 0.02, abs_eps: float | None = None,
                         min_keep: int = 1, max_keep: int | None = None) -> list[str]:
    """
    最良MAEに近いモデルのみ採用
      - rel_delta: 相対閾値（best * (1 + rel_delta) まで許容）
      - abs_eps:   絶対差の閾値（best + abs_eps まで許容、指定時は両方満たす）
    """
    if len(mae_s) == 0 or not np.isfinite(mae_s.iloc[0]):
        return []
    best = mae_s.iloc[0]
    keep = []
    for name, v in mae_s.items():
        ok_rel = (v <= best * (1.0 + rel_delta))
        ok_abs = True if abs_eps is None else (v <= best + abs_eps)
        if ok_rel and ok_abs:
            keep.append(name)

    if len(keep) < min_keep:
        keep = [mae_s.index[0]]
    if max_keep is not None and len(keep) > max_keep:
        keep = keep[:max_keep]
    return keep
    
def as_nan(x):
    # 数値で、番兵値と一致するものを NaN に
    if pd.isna(x):
        return np.nan
    try:
        return np.nan if float(x) == float(NULL_FOR_SUBMISSION) else x
    except Exception:
        return x

def prepare_oof_table(train, dfs, target):
    """
    train: 学習テーブル（id, target を含む）
    dfs:   {model_name: oof_df (id, target を含む)} の辞書
    target: 例 'Tg'
    戻り値: df (列: ['id','y_true', <model1>, <model2>, ...])
    """
    base = train[['id', target]].dropna().rename(columns={target: 'y_true'}).copy()
    for name, df in dfs.items():
        # id で inner merge（id順を base に合わせて固定）
        base = base.merge(df[['id', target]].rename(columns={target: name}),
                          on='id', how='left')
    return base

def rank_average(df_models: pd.DataFrame) -> np.ndarray:
    Z = df_models.loc[:, df_models.notna().any(axis=0)]
    if Z.shape[1] == 0:
        return np.full(len(df_models), np.nan)
    R = Z.rank(method='average', na_option='keep', pct=True)
    return R.mean(axis=1, skipna=True).to_numpy()

def mean_blend(df_models: pd.DataFrame) -> np.ndarray:
    Z = df_models.loc[:, df_models.notna().any(axis=0)]
    if Z.shape[1] == 0:
        return np.full(len(df_models), np.nan)
    return Z.mean(axis=1, skipna=True).to_numpy()

def median_blend(df_models: pd.DataFrame) -> np.ndarray:
    Z = df_models.loc[:, df_models.notna().any(axis=0)]
    if Z.shape[1] == 0:
        return np.full(len(df_models), np.nan)
    return Z.median(axis=1, skipna=True).to_numpy()

def nnls_blend(df_models: pd.DataFrame, y: np.ndarray):
    Z = df_models.loc[:, df_models.notna().any(axis=0)]
    M = Z.columns.tolist()
    if len(M) == 0:
        return np.full(len(df_models), np.nan), np.array([])

    # 学習は「全モデル非NaN」かつ y が NaN でない行のみ
    mask_fit = Z.notna().all(axis=1).values & ~np.isnan(y)
    if mask_fit.sum() < max(20, len(M)):  # 行が少なすぎたら平均にフォールバック
        return mean_blend(df_models), np.ones(len(M))/len(M)

    X_fit = Z.loc[mask_fit, M].to_numpy()
    y_fit = y[mask_fit]
    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(X_fit, y_fit)
    w = reg.coef_.copy()
    w = np.maximum(w, 0)
    s = w.sum()
    w = w / s if s > 0 else np.ones_like(w)/len(w)

    # 推論：行ごとに利用可能な列の重みだけで再正規化
    X_all = Z.to_numpy()
    pred = np.full(len(df_models), np.nan)
    for i in range(len(df_models)):
        row = X_all[i, :]
        avail = ~np.isnan(row)
        if not avail.any():
            continue
        ww = w[avail]
        xx = row[avail]
        s = ww.sum()
        pred[i] = (ww @ xx) / s if s > 0 else np.nanmean(xx)
    return pred, list(w)
def nnls_positive_blend(X_sel: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    選抜済み列 X_sel について、非負・総和1の重みで OOF を線形結合
    学習: すべての選抜列が非NaN & y 非NaN の行のみ
    推論: 行ごとに利用可能列で重みを再正規化
    """
    if X_sel.shape[1] == 0:
        return np.full(len(y), np.nan), np.array([])

    mask_fit = (~np.isnan(y)) & X_sel.notna().all(axis=1).values
    if mask_fit.sum() < max(20, X_sel.shape[1]):  # データが少なすぎる場合は平均へフォールバック
        pred = X_sel.mean(axis=1, skipna=True).to_numpy()
        w = np.ones(X_sel.shape[1]) / X_sel.shape[1]
        return pred, w

    X_fit = X_sel.loc[mask_fit].to_numpy()
    y_fit = y[mask_fit].astype(float)

    reg = Ridge(alpha=1e-6, fit_intercept=False, positive=True)
    reg.fit(X_fit, y_fit)
    w = np.maximum(reg.coef_.copy(), 0.0)
    s = w.sum()
    w = (w / s) if s > 0 else np.ones_like(w) / len(w)

    X_all = X_sel.to_numpy()
    pred = np.full(len(y), np.nan)
    for i in range(len(y)):
        row = X_all[i, :]
        avail = ~np.isnan(row)
        if not avail.any():
            continue
        ww = w[avail]
        xx = row[avail]
        s = ww.sum()
        pred[i] = (ww @ xx) / s if s > 0 else np.nanmean(xx)
    return pred, w
# ---------------------------------------------------------------------------

def blend_per_target_safe(train, dfs_by_model, targets, method="nnls",
                          rel_delta=0.02, abs_eps=None, max_keep=None, verbose=True):
    """
    変更点:
      - 各ターゲットで '最良MAEに近い' モデルだけ選抜してからブレンド
      - 選ばれなかったモデルの重みは 0
      - id で突合（OOFは id がなければ SMILES→id を補完）
    """
    outputs = {}
    tgt_w = {}

    for tgt in targets:
        df = prepare_oof_table(train, dfs_by_model, tgt)  # ['id','y_true', m1, m2,...]
        y = df['y_true'].to_numpy().astype(float)
        X = df.drop(columns=['id','y_true']).astype(float)

        # 評価可能行
        avail_row = X.notna().any(axis=1).values & ~np.isnan(y)
        if avail_row.sum() == 0:
            if verbose:
                print(f"[{tgt}] No evaluable rows (y or all models are NaN).")
            outputs[tgt] = pd.DataFrame({"id": df["id"], tgt: np.full(len(df), np.nan)})
            tgt_w[tgt] = {m: 0.0 for m in X.columns}
            continue

        # --- ここが新規：モデル選抜 ---
        mae_s = per_model_mae_table(X, y)
        keep = select_models_by_mae(mae_s, rel_delta=rel_delta, abs_eps=abs_eps,
                                    min_keep=1, max_keep=max_keep)
        if verbose:
            print(f"[{tgt}] best MAE by model:\n{mae_s.head(len(mae_s)).round(6)}")
            print(f"[{tgt}] selected: {keep}")

        X_sel = X[keep]

        # --- ブレンド ---
        if method in ("nnls", "ridge"):
            pred_full, w_sel = nnls_positive_blend(X_sel, y)
            # 全モデルに対応する重み辞書（未採用は 0）
            w_all = {m: 0.0 for m in X.columns}
            for i, m in enumerate(keep):
                w_all[m] = float(w_sel[i])
            if verbose:
                ws = ", ".join([f"{m}:{w_all[m]:.3f}" for m in keep])
                print(f"[{tgt}] weights -> {ws}")
        elif method == "mean":
            pred_full = X_sel.mean(axis=1, skipna=True).to_numpy()
            w_all = {m: (1.0/len(keep) if m in keep else 0.0) for m in X.columns}
        elif method == "median":
            pred_full = X_sel.median(axis=1, skipna=True).to_numpy()
            w_all = {m: 0.0 for m in X.columns}  # 参考値：中央値なので重みを出さない
        elif method == "rank_avg":
            R = X_sel.rank(method='average', na_option='keep', pct=True)
            pred_full = R.mean(axis=1, skipna=True).to_numpy()
            w_all = {m: (1.0/len(keep) if m in keep else 0.0) for m in X.columns}
        else:
            raise ValueError("unknown method")

        # OOF評価（参考）
        mask_eval = avail_row & ~np.isnan(pred_full)
        mae = np.nan if mask_eval.sum() == 0 else mean_absolute_error(y[mask_eval], pred_full[mask_eval])
        if verbose:
            print(f"[{tgt}] OOF MAE: {mae:.6f} (n={mask_eval.sum()})")

        outputs[tgt] = pd.DataFrame({"id": df["id"], tgt: pred_full})
        tgt_w[tgt] = w_all

    return outputs, tgt_w

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", type=str, nargs="*", help="アンサンブルするexp")
    parser.add_argument("-dir", type=str, default="20250830", help="outputs/ensemble 配下に保存するディレクトリ名")
    parser.add_argument("-upload_kaggle", action="store_true")
    parser.add_argument("-rel_delta", type=float, default=0.02)

    return parser.parse_args()

def main():
    targets = ["Tg", "Tc", "Rg", "FFV", "Density"]

    args = get_args()
    if len(args.exp) == 0:
        print("Not exists exp.")
        return
    
    
    train, _ = load_data(abspath / "data/raw")
    output_dir = abspath / "outputs"

    save_dir = output_dir / "ensemble" / args.dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ここ差し替え：SMILES ではなく id で揃える（SMILESしかない場合だけ補完）
    dfs = {}
    for exp in args.exp:
        oof_df = pd.read_csv(output_dir / exp / "oof.csv")
        if "id" not in oof_df.columns:
            # 古い形式: SMILES→id を補完
            tmp = train[["id","SMILES"]].merge(oof_df, how="left", on="SMILES")
            oof_df = tmp.drop(columns=["SMILES"])
        # id ベースで保持
        dfs[exp] = oof_df[["id"] + targets].copy()

    for target in targets:
        train[f"org_{target}"] = train[target]

    train["id"] = np.arange(len(train))

    # 相対閾値は例: 0.02 (=最良MAEの+2%以内のみ採用)
    outs, w = blend_per_target_safe(
        train=train,
        dfs_by_model=dfs,
        targets=targets,
        method="nnls",         # mean/median/rank_avg も可
        rel_delta=args.rel_delta,        # <-- ここを 0.01~0.05 で調整
        abs_eps=None,          # 併用したいなら e.g. abs_eps=0.05
        max_keep=None,         # 上限を設けたいときは整数指定（例: 3）
        verbose=True
    )
    weight_w = {}

    for target in targets:
        weight_w[target] = {}
        for idx, exp in enumerate(args.exp):
            weight_w[target][exp] = w[target][exp]

    print(weight_w)

        
    with open(save_dir / "weights.json", "w") as f:
        json.dump(weight_w, f)
    
    oof_df = pd.DataFrame({
        "id": range(len(train))
    })

    for target in targets:
        df = outs[target]
        oof_df = oof_df.merge(df, how="left", on="id")

    results = vars(args)

    # CV 計算
    solution = train[["id"] + targets].copy()

    # 評価
    final_score = score(
        solution=solution,
        submission=oof_df,
    )
    print(f"\n📊 Final OOF Score (wMAE): {final_score:.6f}")
    results["score"] = round(final_score, 5)
    results["mae"] = {}

    for target in targets:
        solution.loc[solution[target] == NULL_FOR_SUBMISSION, target] = np.nan
        oof_df.loc[oof_df[target] == NULL_FOR_SUBMISSION, target] = np.nan
        y = solution[[target]].dropna()[target]
        y_pred = oof_df[oof_df["id"].isin(solution[["id", target]].dropna().index)][target]
        mae = mean_absolute_error(y, y_pred)
        print(f"{target} mae: {mae:.6f}")
        results["mae"][target] = round(mae, 5)

    print(results)
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f)


    if args.upload_kaggle:
        dataset_title = f"ensemble-weight-{args.dir}"
        dataset_id = f"koya346/{dataset_title}"
        create_kaggle_dataset_metadata(dataset_title, dataset_id, save_dir)
        upload_kaggle_dataset(dataset_id, save_dir)
    

    return 

if __name__ == "__main__":
    main()
