from rdkit import Chem

def generate_trimer(smiles: str, cyclic: bool = False) -> str:
    # Parse repeat unit and atom maps
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    dummies = [a for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    if len(dummies) != 2:
        raise ValueError("SMILES must contain exactly two '*' points.")
    
    # Map attachment atoms
    for i, dummy in enumerate(dummies):
        neigh = dummy.GetNeighbors()[0]
        neigh.SetAtomMapNum(100 + i)

    # Remove dummy atoms
    em = Chem.EditableMol(mol)
    for atom in sorted(dummies, key=lambda a: a.GetIdx(), reverse=True):
        em.RemoveAtom(atom.GetIdx())
    core = em.GetMol()
    Chem.SanitizeMol(core, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    # Identify attach indices
    attach = {}
    for atom in core.GetAtoms():
        m = atom.GetAtomMapNum()
        if m == 100:
            attach["start"] = atom.GetIdx()
        elif m == 101:
            attach["end"] = atom.GetIdx()
        atom.SetAtomMapNum(0)
    if "start" not in attach or "end" not in attach:
        raise ValueError("Attachment points not found.")
    
    # Combine three copies
    n = core.GetNumAtoms()
    combo = core
    for _ in range(2):
        combo = Chem.CombineMols(combo, core)
    rw = Chem.RWMol(combo)

    # Connect linear bonds
    for i in range(2):
        a = attach["end"] + i * n
        b = attach["start"] + (i + 1) * n
        if rw.GetBondBetweenAtoms(a, b) is None:
            rw.AddBond(a, b, Chem.BondType.SINGLE)

    if cyclic:
        a = attach["end"] + 2 * n
        b = attach["start"]
        if rw.GetBondBetweenAtoms(a, b) is None:
            rw.AddBond(a, b, Chem.BondType.SINGLE)

    mol_trim = rw.GetMol()
    Chem.SanitizeMol(mol_trim, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    return Chem.MolToSmiles(mol_trim, kekuleSmiles=False)
