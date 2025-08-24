import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MolFromSmiles, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from mordred import Calculator, descriptors


def add_maccs(df):
    maccs_arr = np.zeros((len(df), 167), dtype=np.int8)
    for idx, smiles in enumerate(tqdm(df["SMILES"].values, desc="Generating maccs")):
        mol = MolFromSmiles(smiles)
        maccs_arr[idx, :] = MACCSkeys.GenMACCSKeys(mol)
    maccs_df = pd.DataFrame(maccs_arr)
    maccs_df.columns = [f"maccs_{idx}" for idx in range(maccs_df.shape[1])]
    return pd.concat([df, maccs_df], axis=1)



def get_mfp(mol, radius, fp_size):
    if mol is None:
        return np.zeros((fp_size,), dtype=np.int8)
    generator = GetMorganGenerator(radius=radius, fpSize=fp_size)
    return generator.GetFingerprint(mol)
    
def add_descriptors(df, radius=2, fp_size=1024):
    descriptor_names  = [name for (name, _) in Descriptors.descList]
    descs = []
    mfp_mat = np.empty((len(df), fp_size), dtype=np.int8)
    
    for idx, smi in enumerate(tqdm(df["SMILES"], desc="Generating descriptors")):
        mol = MolFromSmiles(smi)
        # rdkit desc (失敗したら nan)
        row = []
        for _, func in Descriptors.descList:
            try:
                row.append(func(mol))
            except Exception:
                row.append(np.nan)
        descs.append(row)
        mfp_mat[idx] = get_mfp(mol=mol, radius=radius, fp_size=fp_size)
        
    desc_df = pd.DataFrame(descs, columns=descriptor_names)
    mfp_df = pd.DataFrame(mfp_mat, columns=[f"mfp_vec{i}" for i in range(fp_size)])
    out = pd.concat([df, desc_df, mfp_df], axis=1).reset_index(drop=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
    

def replace_dummy_with_h(mol):
    """
    * を水素に置換する。
    3D 配座の最適化の際にダミーデータがあると困る
    """
    rw = Chem.RWMol(mol)
    for atom in list(mol.GetAtoms()):
        if atom.GetSymbol() == "*":
            rw.ReplaceAtom(atom.GetIdx(), Chem.Atom("H"))
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol

def generate_conform_3d(mol, n=10, max_iters=200):
    """
    安定な立体配座を計算し、最低エネルギー配座だけ残して返す
    """
    if mol is None:
        return None

    mol = replace_dummy_with_h(mol)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.numThreads = 0 # 0: 全スレッド
    params.randomSeed = 42 # 再現性
    cid_list = AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params)
    if len(cid_list) == 0:
        # 埋め込み失敗時は H 対か前に戻す
        mol = Chem.RemoveHs(mol, sanitize=True)
        return mol
    
    # MMFF で最適化(可能なら MMFF, ダメならUFF)
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94s", maxIters=max_iters)
        energies = [e for (_, e) in res]
    except Exception:
        res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
        energies = [e for (_, e) in res]

    # 最小エネルギー配座を選び、それ以外を削除
    best_idx = int(np.argmin(energies))
    best_cid = cid_list[best_idx]
    conf_ids = [int(c) for c in cid_list]
    for cid in conf_ids:
        if cid != best_cid:
            mol.RemoveConformer(cid)

    return mol
    
def add_descriptors_mordred(df, num_confs=3, ignore_3D=True, ignore_3d_stats=True):
    """
    Mordred 記述子を返す
    - 3D は最低エネルギー配座1つに絞る
    - 失敗時は2Dで代替
    """
    calc = Calculator(descriptors, ignore_3D=ignore_3D)
    mols = []

    features_3d = []

    for smi in tqdm(df["SMILES"].values, desc="mordred desc"):
        try:
            base = Chem.MolFromSmiles(smi)
            mol = generate_conform_3d(base, n=num_confs) if not ignore_3D else base
            if not ignore_3d_stats:
                feats = get_3d_summary_stats(mol)
                features_3d.append(feats)
                # 3次元特徴を計算
            if mol is None:
                mol = base
            mols.append(mol)
        except Exception as e:
            print(f"{smi}, {e}")
            mol = Chem.MolFromSmiles(smi)
            mols.append(mol) 

    desc_df = calc.pandas(mols)

    # 数値化, 例外値処理
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)
    df = pd.concat([df, desc_df], axis=1)
    if len(features_3d):
        df = pd.concat([df, pd.DataFrame(features_3d)], axis=1)
    return df.reset_index(drop=True)


def get_3d_summary_stats(mol):
    """
    mol: 3D座標をもつ RDKit Mol (単一コンフォーマ)
    return: dict 形式の統計特徴
    """
    if mol is None or mol.GetNumConformers() == 0:
        return {
            "dist_mean": np.nan,
            "dist_std": np.nan,
            "dist_p10": np.nan,
            "dist_p90": np.nan,
            "angle_mean": np.nan,
            "angle_std": np.nan,
            "n_contacts_3A": np.nan,
            "n_contacts_5A": np.nan,
    }
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    # 距離行列
    dists = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
            dists.append(dist)
    dists = np.array(dists)
    dist_mean = np.mean(dists)
    dist_std = np.std(dists)
    dist_p10 = np.percentile(dists, 10)
    dist_p90 = np.percentile(dists, 90)

    # 角度統計
    angles = []
    for i in range(n_atoms):
        neighs = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(i).GetNeighbors()]
        for a in range(len(neighs)):
            for b in range(a+1, len(neighs)):
                v1 = np.array(conf.GetAtomPosition(neighs[a])) - np.array(conf.GetAtomPosition(i))
                v2 = np.array(conf.GetAtomPosition(neighs[b])) - np.array(conf.GetAtomPosition(i))
                cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
                angles.append(angle)
    angles = np.array(angles) if angles else np.array([np.nan])
    angles_mean = np.nanmean(angles)
    angles_std = np.nanstd(angles)

    # 近接カウント
    n_contacts_3A = np.sum(dists < 3.0)
    n_contacts_5A = np.sum(dists < 5.0)
    return {
        "dist_mean": dist_mean, 
        "dist_std": dist_std, 
        "dist_p10": dist_p10, 
        "dist_p90": dist_p90,
        "angle_mean": angles_mean, 
        "angle_std": angles_std, 
        "n_contacts_3A": n_contacts_3A,
        "n_contacts_5A": n_contacts_5A, 
    }





