import numpy as np
import pandas as pd

from .canonical import make_smile_canonical


def add_external_data(df, ex_path, col, rename_d=None, is_complement=True):
    """
    - df に外部データを追加
    - 新たに追加された外部データにフラグ付与
    - is_complement=True の場合はdf の target が null のとき外部データで補完する
    """
    ex_df = pd.DataFrame()
    if ".csv" in str(ex_path):
        ex_df = pd.read_csv(ex_path)
    elif ".xlsx" in str(ex_path):
        ex_df = pd.read_excel(ex_path)
    else:
        ValueError(f"Unexpected extension {ex_path}")

    if rename_d is not None:
        ex_df = ex_df.rename(columns=rename_d)

    # smiles-extra-data/data_tg3.xlsx の Tg はケルビン
    if "smiles-extra-data/data_tg3.xlsx" in str(ex_path):
        ex_df[col] -= 273.15

    if "smiles-extra-data/data_dnst1.xlsx" in str(ex_path):
        ex_df = ex_df[ex_df[col]!="nylon"]
        ex_df[col] = ex_df[col].astype(float)
        ex_df[col] -= 0.118

    ex_df["SMILES"] = ex_df["SMILES"].apply(make_smile_canonical)
    ex_df = ex_df.groupby("SMILES")[col].mean().reset_index()
    ex_df = ex_df[["SMILES", col]]

    if is_complement:
        ex_df.columns = ["SMILES", f"{col}_ex"]
        # df に含まれている SMILES
        df = df.merge(ex_df, how="left", on="SMILES")
        # df に含まれる値を優先し、なければ追加データの値を参照する
        df[col] = np.where(df[col].notnull(), df[col], df[f"{col}_ex"])
        df = df.drop([f"{col}_ex"], axis=1)

        # df に含まれていない SMILES
        cond = ~ex_df["SMILES"].isin(df["SMILES"].values)
        ex_df.columns = ["SMILES", col]
        df = pd.concat([df, ex_df[cond]]).reset_index(drop=True)
        
    else:
        # 単純に結合する
        df = pd.concat([df, ex_df]).reset_index(drop=True)

    return df
