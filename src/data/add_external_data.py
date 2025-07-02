from pathlib import Path
import numpy as np
import pandas as pd

def add_external_data(train: pd.DataFrame, data_path: Path):
    """
    学習データに外部データを突合する
    """
    ex_tg_df = pd.read_csv(data_path / "smiles-tg/Tg_SMILES_class_pid_polyinfo_median.csv")[["SMILES", "Tg"]]
    ex_tc_df = pd.read_csv(data_path / "tc-smiles/Tc_SMILES.csv")
    ex_tc_df.columns = ["Tc", "SMILES"]

    # スコア悪化
    # train_merged = pd.concat([train, ex_tg_df, ex_tc_df])
    
    # ex データを SMILES 毎に一意化
    ex_tg_df = ex_tg_df.groupby("SMILES")["Tg"].min().reset_index()
    ex_tc_df = ex_tc_df.groupby("SMILES")["Tc"].min().reset_index()
    
    # Tg, Tc 外部データ
    train_merged = train.merge(ex_tg_df, how="left", on="SMILES", suffixes=("", "_ex"))
    train_merged = train_merged.merge(ex_tc_df, how="left", on="SMILES", suffixes=("", "_ex"))
    # train_merged["org_Tg"] = train_merged["Tg"]
    # train_merged["org_Tc"] = train_merged["Tc"]
    
    # 元の train の値を優先
    train_merged["Tg"] = np.where(~train_merged["Tg"].isnull(), train_merged["Tg"], train_merged["Tg_ex"])
    train_merged["Tc"] = np.where(~train_merged["Tc"].isnull(), train_merged["Tc"], train_merged["Tc_ex"])
    
    # train に含まれていなければ concat で追加する
    cond_include_ex_tg = train["SMILES"].isin(ex_tg_df["SMILES"].values)
    cond_include_ex_tc = train["SMILES"].isin(ex_tc_df["SMILES"].values)
    
    train_merged = pd.concat([train_merged, ex_tg_df[~cond_include_ex_tg]])
    train_merged = pd.concat([train_merged, ex_tc_df[~cond_include_ex_tc]])
    train_merged.drop(["Tg_ex", "Tc_ex"], axis=1, inplace=True)

    return train_merged.reset_index(drop=True) 

