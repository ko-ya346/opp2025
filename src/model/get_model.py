from .torch_molecule import (
    GNNMolecularPredictor, 
    GREAMolecularPredictor, 
    SGIRMolecularPredictor, 
    SMILESTransformerMolecularPredictor, 
    BFGNNMolecularPredictor, 
    DIRMolecularPredictor,
    GRINMolecularPredictor,
    IRMMolecularPredictor,
    LSTMMolecularPredictor,
    RPGNNMolecularPredictor,
    SSRMolecularPredictor,
)

def get_model(model_name: str):
    model_dict = {
        "gnn": GNNMolecularPredictor,
        "grea": GREAMolecularPredictor,
        "sgir": SGIRMolecularPredictor,
        "smiles_transformer": SMILESTransformerMolecularPredictor,
        "bfgnn": BFGNNMolecularPredictor,
        "dir": DIRMolecularPredictor,
        "grin": GRINMolecularPredictor,
        "irm": IRMMolecularPredictor,
        "lstm": LSTMMolecularPredictor,
        "rpgnn": RPGNNMolecularPredictor,
        "ssr": SSRMolecularPredictor,
    }

    if model_name not in model_dict:
        ValueError(f"{model_name} does not exist.")

    return model_dict[model_name]

