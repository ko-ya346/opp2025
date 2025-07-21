from rdkit import Chem

import numpy as np

def make_smile_canonical(smile):
    """
    Converts a SMILES string to its canonical form to ensure uniqueness.
    Returns np.nan if conversion fails.
    """
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return np.nan
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except Exception:
        return np.nan
