from rdkit import Chem
from rdkit.Chem import rdchem

def _is_aromatic_n(atom: rdchem.Atom) -> bool:
    return atom.GetAtomicNum() == 7 and atom.GetIsAromatic()

def _is_nh(atom: rdchem.Atom) -> bool:
    return (atom.GetTotalNumHs() > 0)

def _prepare_aromatic_n_for_attachment(rw: Chem.RWMol, idx: int, mode: str):
    """
    芳香族Nへの結合前処理
      - allow_h_loss: [nH] の H を1つ削って中性の n-R にしてから結合
      - quaternize: 中性n を [n+] にしてから結合(第四級化)
      - forbid: 例外
    """
    a = rw.GetAtomWithIdx(idx)
    if not _is_aromatic_n(a):
        return 

    if mode == "forbid":
        raise ValueError("Attachment to aromatic 'n' is forbidden (set aromatic_n_mode to 'allow_h_loss' or 'quaternize').")
    if mode == "allow_h_loss":
        if not _is_nh(a):
            raise ValueError("Aromatic 'n' has no H to replace; use aromatic_n_mode='quaternize'.")
        if a.GetNumExplicitHs() == 0:
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(1)
        a.SetNumExplicitHs(a.GetNumExplicitHs() - 1)
        a.UpdatePropertyCache(strict=False)
        return
    if mode == "quaternize":
        if a.GetFormalCharge() != 1:
            a.SetFormalCharge(1)
            a.SetNoImplicit(True)
            a.UpdatePropertyCache(strict=False)
        return
    raise ValueError(f"Unknown aromatic_n_mode: {mode}")

def generate_trimer(smiles: str, cyclic: bool = False, aromatic_n_mode: str = "forbid") -> str:
    # Parse repeat unit and atom maps
    mol0 = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol0 is None:
        raise ValueError(f"Invalid SMILES (parse failed): {smiles}")
    Chem.SanitizeMol(mol0)

    dummies = [a for a in mol0.GetAtoms() if a.GetSymbol() == "*"]
    if len(dummies) != 2:
        raise ValueError("SMILES must contain exactly two '*' points.")
    
    # Map attachment atoms
    for i, d in enumerate(dummies):
        nbs = d.GetNeighbors()
        if len(nbs) != 1:
            raise ValueError(f"Dummy at idx={d.GetIdx()} must have exactly one neighbor.")
        nbs[0].SetAtomMapNum(100 + i)

    # Remove dummy atoms
    em = Chem.EditableMol(mol0)
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
        if m in (100, 101):
            atom.SetAtomMapNum(0)
    if "start" not in attach or "end" not in attach:
        raise ValueError("Attachment points not found (map 100/101).")
    
    # Combine three copies
    n = core.GetNumAtoms()
    combo = core
    for _ in range(2):
        combo = Chem.CombineMols(combo, core)
    rw = Chem.RWMol(combo)

    def _add_bond_safely(a_idx: int, b_idx: int, tag: str):
        """
        芳香族 n の場合は事前処理
        """
        _prepare_aromatic_n_for_attachment(rw, a_idx, aromatic_n_mode)
        _prepare_aromatic_n_for_attachment(rw, b_idx, aromatic_n_mode)

        A = rw.GetAtomWithIdx(a_idx)
        B = rw.GetAtomWithIdx(b_idx)
        A.UpdatePropertyCache(strict=False) 
        B.UpdatePropertyCache(strict=False) 
        if rw.GetBondBetweenAtoms(a_idx, b_idx) is None:
            rw.AddBond(a_idx, b_idx, Chem.BondType.SINGLE)
        # 追加後にサニタイズして早期検出
        try:
            _ = rw.GetMol()
            Chem.SanitizeMol(_,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES | 
                             Chem.SanitizeFlags.SANITIZE_SYMMRINGS | 
                             Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            )
        except Exception as e:
            raise ValueError(f"Failed after {tag} bond ({a_idx}-{b_idx}): {e}")

    # Connect linear bonds
    for i in range(2):
        a = attach["end"] + i * n
        b = attach["start"] + (i + 1) * n
        _add_bond_safely(a, b, tag=f"linear[{i}]")

    if cyclic:
        a = attach["end"] + 2 * n
        b = attach["start"]
        _add_bond_safely(a, b, tag="cyclic")

    mol_trim = rw.GetMol()
    Chem.SanitizeMol(mol_trim)

    return Chem.MolToSmiles(mol_trim, kekuleSmiles=False)
