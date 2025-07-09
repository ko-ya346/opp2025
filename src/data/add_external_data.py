import numpy as np
import pandas as pd


def add_external_data(df, ex_path, col, rename_d=None):
    ex_df = pd.read_csv(ex_path)
    if rename_d is not None:
        ex_df = ex_df.rename(columns=rename_d)
    ex_df = ex_df.groupby("SMILES")[col].mean().reset_index()
    ex_df = ex_df[["SMILES", col]]
    ex_df.columns = ["SMILES", f"{col}_ex"]

    # df に含まれている SMILES
    df = df.merge(ex_df, how="left", on="SMILES")
    # df に含まれていない SMILES
    cond = ~ex_df["SMILES"].isin(df["SMILES"].values)
    df = pd.concat([df, ex_df[cond]]).reset_index(drop=True)
    
    # df に含まれる値を優先し、なければ追加データの値を参照する
    df[col] = np.where(df[col].notnull(), df[col], df[f"{col}_ex"])
    df = df.drop([f"{col}_ex"], axis=1)
    return df
