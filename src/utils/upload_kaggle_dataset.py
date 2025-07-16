import os
from kaggle.api.kaggle_api_extended import KaggleApi
import json
import subprocess
from pathlib import Path

os.environ["KAGGLE_CONFIG_DIR"] = str(Path.home() / ".kaggle")


def create_kaggle_dataset_metadata(title: str, dataset_id: str, dir):
    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}]
    }

    (dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=4))

def upload_kaggle_dataset(dataset_id: str, dir):
    api = KaggleApi()
    api.authenticate()

    dataset_exists = False

    try:
        api.dataset_status(dataset_id)

        dataset_exists = True
        print(f"✅ Dataset '{dataset_id}' exists. Updating...")
    except Exception:
        print(f"ℹ️ Dataset '{dataset_id}' does not exist. Creating...")
        print()

    try:
        if dataset_exists:
            subprocess.run([
                "kaggle", "datasets", "version",
                "-p", str(dir),
                "-m", f"Update for {dataset_id}",
                "--dir-mode", "zip"
            ])
        else:
            subprocess.run([
                "kaggle", "datasets", "create",
                "-p", str(dir),
                "--dir-mode", "zip"
            ])
    except:
        print("Upload failed.")
