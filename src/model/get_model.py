from .torch_molecule import GNNMolecularPredictor, GREAMolecularPredictor, SGIRMolecularPredictor

def get_model(model_name: str):
    model_dict = {
        "gnn": GNNMolecularPredictor,
        "grea": GREAMolecularPredictor,
        "sgir": SGIRMolecularPredictor,
    }

    if model_name not in model_dict:
        ValueError(f"{model_name} does not exist.")

    return model_dict[model_name]

