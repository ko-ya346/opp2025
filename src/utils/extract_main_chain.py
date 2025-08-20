from rdkit import Chem
from rdkit.Chem import rdmolops

def extract_main_chain_smiles_from_star(
    smiles: str,
    expand_rings: bool = True,
    ring_overlap_threshold: int = 2,
    expand_ring_systems: bool = True,
    preserve_aromatics: bool = True,   # ベンゼンの二重結合を残したい場合 True のまま
) -> str:
    """
    '*' で両端結合点を示す繰返し単位 SMILES から主鎖（*の隣接原子間の最短パス）を抽出し SMILES を返す。
    - expand_rings: パスが通るリングをまるごと含める
    - ring_overlap_threshold: パスがそのリング上を通る最小原子数（>=2 推奨）
    - expand_ring_systems: 選ばれたリングに融合するリング群も含める
    - preserve_aromatics: 芳香環を含めてフルサニタイズ（Kekulize）を通し、二重結合を残す
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # 1) '*' を探し、各 '*' の隣接原子 idx を取得（ここでは分子をいじらない）
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    if len(stars) != 2:
        raise ValueError("SMILES must contain exactly two '*' atoms.")

    star_neighbors = []
    for s_idx in stars:
        s_atom = mol.GetAtomWithIdx(s_idx)
        nbs = s_atom.GetNeighbors()
        if len(nbs) != 1:
            raise ValueError(f"Each '*' must have exactly one neighbor (star idx={s_idx}).")
        star_neighbors.append(nbs[0].GetIdx())

    a0, a1 = star_neighbors[0], star_neighbors[1]

    # 2) 最短パス（元の分子上）
    path_atoms = set(rdmolops.GetShortestPath(mol, a0, a1))  # set of atom indices

    # 3) 条件付きでリングを取り込む（元の分子のリング情報を使う）
    if expand_rings:
        ri = mol.GetRingInfo()
        atom_rings = list(ri.AtomRings())  # 各リングは原子 idx のタプル
        selected = set()
        for i, ring in enumerate(atom_rings):
            overlap = sum(1 for x in ring if x in path_atoms)
            if overlap >= ring_overlap_threshold:
                selected.add(i)

        if expand_ring_systems and selected:
            ring_sets = [set(r) for r in atom_rings]
            changed = True
            while changed:
                changed = False
                union_atoms = set().union(*(ring_sets[i] for i in selected))
                for j, rs in enumerate(ring_sets):
                    if j in selected:
                        continue
                    if union_atoms & rs:  # 原子共有で融合とみなす
                        selected.add(j)
                        changed = True

        for i in selected:
            path_atoms.update(atom_rings[i])

    # 4) サブグラフを作成：保持する原子 = path_atoms、かつ '*' は除外
    keep = path_atoms.copy()
    keep.difference_update(stars)  # '*' 自体は除く
    if not keep:
        # 両方の '*' が同じ原子を指すなどで経路長0 → その原子を主鎖とする
        # （この場合、star_neighbors は同一原子）
        keep.add(a0)

    # 5) 余計な原子を削除してサブモル
    rw = Chem.RWMol(mol)
    drop = sorted(set(range(mol.GetNumAtoms())) - keep, reverse=True)
    for idx in drop:
        rw.RemoveAtom(idx)
    sub = rw.GetMol()

    # 6) サニタイズ（まずはフルでトライ：ベンゼンの二重結合を残す）
    def _try_full_sanitize(m):
        Chem.SanitizeMol(m)  # フル（Kekulize+芳香性設定 含む）
        return m

    def _sanitize_without_kekule(m):
        # 最終手段：芳香性を剥がし、Kekulize/SetAromaticity をスキップ
        for a in m.GetAtoms():
            if a.GetIsAromatic():
                a.SetIsAromatic(False)
        for b in m.GetBonds():
            if b.GetIsAromatic() or b.GetBondType() == Chem.BondType.AROMATIC:
                b.SetIsAromatic(False)
                b.SetBondType(Chem.BondType.SINGLE)
        ops = (Chem.SanitizeFlags.SANITIZE_ALL
               ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
               ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        Chem.SanitizeMol(m, sanitizeOps=ops)
        return m

    # 6-1. まずフルを試す
    try:
        _try_full_sanitize(sub)
    except Exception as e1:
        if not preserve_aromatics:
            _sanitize_without_kekule(sub)
        else:
            # 6-2. 再試行：リング取り込みを緩和（閾値=1 & 融合も含める）→ もう一度抽出し直し
            if expand_rings and ring_overlap_threshold > 1:
                return extract_main_chain_smiles_from_star(
                    smiles,
                    expand_rings=True,
                    ring_overlap_threshold=1,
                    expand_ring_systems=True,
                    preserve_aromatics=True,
                )
            # 6-3. それでもダメなら最後に Kekulize スキップで救済（この場合はベンゼンの = は残らない）
            _sanitize_without_kekule(sub)

    return Chem.MolToSmiles(sub, canonical=True, isomericSmiles=False)

