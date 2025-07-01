import torch
import torch.nn.functional as F

from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem


ATOM_LIST = ['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'UNK']
ATOM_MAP = {atom: idx for idx, atom in enumerate(ATOM_LIST)}  # One-hotの位置

def smiles_to_graph(smiles_str: str, y_val=None):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None  # SMILES 文字列が無効な場合

    # --- ノード特徴量（原子） ---
    node_features = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        idx = ATOM_MAP.get(symbol, ATOM_MAP['UNK'])  # 未知は UNK に分類
        one_hot = [0] * len(ATOM_MAP)
        one_hot[idx] = 1

        numeric_features = [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic())
        ]

        features = one_hot + numeric_features
        node_features.append(features)
    x = torch.tensor(node_features, dtype=torch.float)

    # --- エッジインデックスとエッジ特徴量（結合） ---
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()

        # 無向グラフなので両方向追加
        edge_indices += [(i, j), (j, i)]
        edge_attrs += [[bond_type], [bond_type]]
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # --- Dataオブジェクトを返す ---
    data_args = dict(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles_str)
    if y_val is not None:
        data_args["y"] = torch.tensor([y_val], dtype=torch.float)  # shape: (1,)

    return Data(**data_args)

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, targets, transform=None):
        self.smiles_list = smiles_list
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        y = self.targets[idx]
        data = smiles_to_graph(smiles, y)
        if self.transform:
            data = self.transform(data)
        return data


class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GNNModel, self).__init__()
        torch.manual_seed(42)
        
        # GNN layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels * 4)
        
        # Output layer
        self.lin = torch.nn.Linear(hidden_channels * 4, 1)

    def forward(self, data):
        # Unpack data object
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. GNN layers with activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        
        # 2. Global Pooling layer
        x = global_mean_pool(x, batch)
        
        # 3. Final Linear layer for regression
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin(x)
        
        return x
