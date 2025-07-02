import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles
from tqdm.auto import tqdm

def generate_random_smiles(smiles: str, num_augments: int = 3) -> list:
    """
    同じ意味で表記が異なる SMILES を生成
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        return []
    return [MolToSmiles(mol, doRandom=True) for _ in range(num_augments)]

def augment_random_smiles_df(df: pd.DataFrame, num_augments: int = 3) -> pd.DataFrame:
    augmented_rows = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Augmenting SMILES x{num_augments}"):
        smiles = row['SMILES']
        augmented_smiles = generate_random_smiles(smiles, num_augments=num_augments)
        
        for aug_smi in augmented_smiles:
            new_row = row.copy()
            new_row['SMILES'] = aug_smi
            augmented_rows.append(new_row)

    # 元のデータと結合
    augmented_df = pd.DataFrame(augmented_rows)
    return pd.concat([df, augmented_df], ignore_index=True)

