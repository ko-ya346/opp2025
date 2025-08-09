import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MolFromSmiles

from mordred import Calculator, descriptors


def get_mfp(mol, radius, fp_size):
    if mol is None:
        return np.zeros((fp_size,), dtype=np.int8)
    mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
    return np.frombuffer(mfp.ToBitString().encode("ascii"), dtype="S1").astype(np.int8) - ord(b'0')
    
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
    mfp_df = pd.DataFrame(mfp_mat, columns=[f"mfp_vec{i}" for i in range(fp_size)]
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
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94s", maxIters=Max_iters)
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
    
def add_descriptors_mordred(df, num_confs=10, ignore_3D=True):
    """
    Mordred 記述子を返す
    - 3D は最低エネルギー配座1つに絞る
    - 失敗時は2Dで代替
    """
    calc = Calculator(descriptors, ignore_3D=ignore_3D)
    mols = []

    descs = []

    for smi in tqdm(df["SMILES"].values):
        try:
            base = Chem.MolFromSmiles(smi)
            mol = generate_conform_3d(base, n=num_confs) if not ignore_3D else base
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
    return pd.concat([df, desc_df], axis=1).reset_index(drop=True)




