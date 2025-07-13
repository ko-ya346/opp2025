from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def generate_scaffold(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold

