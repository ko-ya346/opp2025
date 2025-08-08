import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MolFromSmiles

from mordred import Calculator, descriptors

# ---------------------------
# 分子記述子を生成する関数
# ---------------------------
def compute_all_descriptors(mol):
    return [desc[1](mol) for desc in Descriptors.descList]



def get_mfp(mol, radius, fp_size):
    if mol is None:
        return np.zeros((1, fp_size))
    mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
    return np.array(list(mfp.ToBitString())).astype(int)
    
def add_descriptors(df, radius=2, fp_size=1024):
    descriptor_names  = [desc[0] for desc in Descriptors.descList]
    descs = []
    mfp_vec = np.empty((len(df), fp_size))
    
    for idx, smi in enumerate(tqdm(df["SMILES"], desc="Generating descriptors")):
        mol = MolFromSmiles(smi)
        descs.append(compute_all_descriptors(mol))
        mfp_vec[idx] = get_mfp(mol=mol, radius=radius, fp_size=fp_size)
        
    desc_df = pd.DataFrame(descs)
    mfp_df = pd.DataFrame(mfp_vec)
    mfp_df.columns = [f"mfp_vec{i}" for i in range(fp_size)]
    df[descriptor_names] = desc_df
    df = pd.concat([df, mfp_df], axis=1).reset_index(drop=True)
    return df
    

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

def generate_conform_3d(mol, n=10):
    """
    安定な立体配座を計算する
    """
    mol = replace_dummy_with_h(mol)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    _ = AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=10)
    return mol
    
def add_descriptors_mordred(df, num_confs=10, ignore_3D=True):
    """
    mordred の記述子を返す
    """
    calc = Calculator(descriptors, ignore_3D=ignore_3D)
    mols = []

    descs = []

    for smi in tqdm(df["SMILES"].values):
        try:
            mol = generate_conform_3d(Chem.MolFromSmiles(smi), n=num_confs)
            mols.append(mol)
        except ValueError as e:
            print(f"{smi}, {e}")
            mol = Chem.MolFromSmiles(smi)
            mols.append(mol) 

        mol = MolFromSmiles(smi)
        descs.append(compute_all_descriptors(mol))

    desc_df = calc.pandas(mols)
    return pd.concat([df, desc_df], axis=1) 




