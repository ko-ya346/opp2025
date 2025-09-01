import networkx as nx
import numpy as np
from rdkit import Chem

def compute_graph_features(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        G = nx.from_numpy_array(adj)
        graph_diameter = nx.diameter(G) if nx.is_connected(G) else 0
        avg_shortest_path = nx.average_shortest_path_length(G) if nx.is_connected(G) else 0
        num_cycles = len(list(nx.cycle_basis(G)))
    except:
        print(f"error: {smile}")
        graph_diameter = 0
        avg_shortest_path = 0
        num_cycles = 0

    return graph_diameter, avg_shortest_path, num_cycles

def add_graph_features(df, col="SMILES"):
    graph_features = np.zeros((len(df), 3))

    for idx, smiles in enumerate(df[col].values):
        graph_diameter, avg_shortest_path, num_cycle = compute_graph_features(smiles)
        graph_features[idx, 0] = graph_diameter
        graph_features[idx, 1] = avg_shortest_path
        graph_features[idx, 2] = num_cycle

    df["graph_diameter"] = graph_features[:, 0]
    df["avg_shortest_path"] = graph_features[:, 1]
    df["num_cycle"] = graph_features[:, 2]
    return df
