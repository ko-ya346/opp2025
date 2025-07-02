from pathlib import Path
import pandas as pd

def load_data(data_dir: Path):
    train = pd.read_csv(data_dir / "neurips-open-polymer-prediction-2025/train.csv")
    test = pd.read_csv(data_dir / "neurips-open-polymer-prediction-2025/test.csv")

    return train, test
