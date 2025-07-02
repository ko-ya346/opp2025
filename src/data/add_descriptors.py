import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from rdkit.Chem import Descriptors, AllChem, MolFromSmiles

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
    
