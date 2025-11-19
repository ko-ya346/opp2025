from rdkit import Chem

def cap_stars_with_H(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        rw = Chem.RWMol(mol)
        for a in list(rw.GetAtoms()):
            if a.GetSymbol() == "*":
                rw.ReplaceAtom(a.GetIdx(), Chem.Atom("H"))
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception as e:
        print(f"{smiles}: {e}")
        return smiles
