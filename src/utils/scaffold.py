import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def generate_scaffold(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold


def add_scaffold_kfold(df, n_splits=5):
    out = df.copy()
    out["scaffold"] = out["SMILES"].apply(generate_scaffold)

    grp = out.groupby("scaffold").size().reset_index(name="count")
    # 大きい骨格から割り当て
    grp = grp.sort_values(["count", "scaffold"], ascending=[False, True]).reset_index(drop=True)
    
    # 同数の骨格が続くときの順序バイアスを抑えるためカルクシャッフル
    same_size_blocks = grp["count"].value_counts().index.tolist()
    grp = grp.sample(frac=1.0, random_state=42).sort_values("count", ascending=False).reset_index(drop=True)

    fold_sizes = [0] * n_splits
    assign = {}

    for idx in range(len(grp)):
        scaff = grp.iloc[idx]["scaffold"]
        cnt = grp.iloc[idx]["count"]
        f = int(np.argmin(fold_sizes))
        assign[scaff] = f
        fold_sizes[f] += cnt

    out["fold"] = out["scaffold"].map(assign).astype(int)

    return out

def scaffold_cv_split(df: pd.DataFrame, fold_col: str = "fold", n_splits: int = 5):
    """
    事前に make_scaffold_folds で fold を振っておく前提。
    これで for fold, (trn_idx, val_idx) in enumerate(scaffold_cv_split(...)) の形で使える。
    """
    assert fold_col in df.columns, f"{fold_col} not found. Run make_scaffold_folds() first."
    for f in range(n_splits):
        val_idx = df.index[df[fold_col] == f].to_numpy()
        trn_idx = df.index[df[fold_col] != f].to_numpy()
        yield f, trn_idx, val_idx
