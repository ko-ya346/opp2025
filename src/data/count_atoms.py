from rdkit import Chem
import pandas as pd

def count_atoms(smile):
    mol = Chem.MolFromSmiles(smile)
    counts = {
        'num_C': 0, 'num_c': 0, 'num_O': 0, 'num_N': 0, 'num_F': 0, 'num_Cl': 0,
        'num_positive_ions': 0, 'num_negative_ions': 0
    }
    if mol is None:
        return counts

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()

        if symbol == 'C':
            if atom.GetIsAromatic():
                counts['num_c'] += 1
            else:
                counts['num_C'] += 1
        elif symbol == 'Cl':
            counts['num_Cl'] += 1
        elif symbol in ['O', 'N', 'F']:
            counts[f'num_{symbol}'] += 1

        if charge > 0:
            counts['num_positive_ions'] += 1
        elif charge < 0:
            counts['num_negative_ions'] += 1

    return counts

def add_count_atoms(df, col="SMILES"):
    count_atom_features = []

    for smiles in df[col].values:
        count_atom_features.append(count_atoms(smiles))

    return pd.concat([df, pd.DataFrame(count_atom_features)], axis=1)

