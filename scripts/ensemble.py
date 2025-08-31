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

def as_nan(x):
    # 数値で、番兵値と一致するものを NaN に
    if pd.isna(x):
        return np.nan
    try:
        return np.nan if float(x) == float(NULL_FOR_SUBMISSION) else x
    except Exception:
        return x

def prepare_oof_table(train, dfs_by_model, target):
    """
    y_true は org_only 列から作るのが超重要！
    """
    org_col = f"org_{target}"
    base = train[['id', org_col]].copy()
    base = base[base[org_col].notnull()]
    base = base.rename(columns={org_col: 'y_true'})

    for name, df in dfs_by_model.items():
        col = df[['id', target]].copy()
        # OOF の番兵値→NaN
        col[target] = col[target].map(as_nan)
        base = base.merge(col.rename(columns={target: name}), on='id', how='left')
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

def blend_per_target_safe(train, dfs_by_model, targets, method="rank_avg", verbose=True):
    outputs = {}
    tgt_w = {}
    for tgt in targets:
        w = []
        df = prepare_oof_table(train, dfs_by_model, tgt)   # ['id','y_true', m1, m2,...]
        y = df['y_true'].to_numpy().astype(float)
        X = df.drop(columns=['id','y_true']).astype(float)

        # 少なくとも1モデルが非NaN & yが非NaN の行のみ評価対象
        avail_row = X.notna().any(axis=1).values & ~np.isnan(y)
        if avail_row.sum() == 0:
            if verbose:
                print(f"[{tgt}] No evaluable rows (y or all models are NaN).")
            outputs[tgt] = pd.DataFrame({"id": df["id"], tgt: np.full(len(df), np.nan)})
            continue

        
        if method == "rank_avg":
            pred_full = rank_average(X)
        elif method == "mean":
            pred_full = mean_blend(X)
        elif method == "median":
            pred_full = median_blend(X)
        elif method == "nnls":
            pred_full, w = nnls_blend(X, y)
            if verbose and len(X.columns) > 0:
                print(f"[{tgt}] NNLS weights:", np.round(w, 3))
        else:
            raise ValueError("unknown method")

        mask_eval = avail_row & ~np.isnan(pred_full)
        if mask_eval.sum() == 0:
            mae = np.nan
        else:
            mae = mean_absolute_error(y[mask_eval], pred_full[mask_eval])
        if verbose:
            print(f"[{tgt}] OOF MAE: {mae:.6f} (n={mask_eval.sum()})")

        outputs[tgt] = pd.DataFrame({"id": df["id"], tgt: pred_full})
        tgt_w[tgt] = w
    return outputs, tgt_w

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", type=str, nargs="*", help="アンサンブルするexp")
    parser.add_argument("-dir", type=str, default="20250830", help="outputs/ensemble 配下に保存するディレクトリ名")
    parser.add_argument("-upload_kaggle", action="store_true")

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

    dfs = {}
    for exp in args.exp:
        df = train[["SMILES"]].copy(deep=True)
        oof_df = pd.read_csv(output_dir / exp / "oof.csv")

        df = df.merge(oof_df, how="left", on="SMILES")
        print(df.head())
        dfs[exp] = df

    for target in targets:
        train[f"org_{target}"] = train[target]

    train["id"] = np.arange(len(train))

    outs, w = blend_per_target_safe(train, dfs, targets, method="nnls")
    weight_w = {}

    for target in targets:
        weight_w[target] = {}
        for idx, exp in enumerate(args.exp):
            weight_w[target][exp] = w[target][idx]

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
