import os
import sys

is_kaggle_notebook = os.path.exists("/kaggle/input")

if is_kaggle_notebook:
    sys.path.append("/kaggle/input/torch-molecule-src/torch-molecule")

from torch_molecule import GNNMolecularPredictor, GREAMolecularPredictor, SGIRMolecularPredictor

def get_model(model_name: str):
    model_dict = {
        "gnn": GNNMolecularPredictor,
        "grea": GREAMolecularPredictor,
        "sgir": SGIRMolecularPredictor,
    }

    if model_name not in model_dict:
        ValueError(f"{model_name} does not exist.")

    return model_dict[model_name]

