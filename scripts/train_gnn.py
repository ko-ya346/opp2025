from pathlib import Path
from glob import glob

import json
import os
import sys
import argparse
import pandas as pd
import numpy as np

import torch

abspath = Path("/home/kouya-takahashi/kaggle/opp2025")
sys.path.append(str(abspath))

from src.model import get_model 
from src.utils.upload_kaggle_dataset import (
    create_kaggle_dataset_metadata,
    upload_kaggle_dataset,
)

def set_seed(SEED=42):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_train", type=str, help="学習データの実験ナンバー")
    parser.add_argument("exp_model", type=str, help="モデルの実験ナンバー")
    parser.add_argument("-model", type=str, default="gnn") 
    parser.add_argument("-n", type=int, default=3, help="異なる seed で学習する回数")
    parser.add_argument("-graph_pooling", choices=["sum", "mean"], default="mean")
    parser.add_argument("-augmented_feature", type=str, nargs="*", help="使用する特徴量. morgan, maccs, desc のいずれか")
    parser.add_argument("-batch_size", type=int, default=256)
    parser.add_argument("-drop_ratio", type=float, default=0.5)
    parser.add_argument("-is_trimmer_cyclic", action="store_true")
    parser.add_argument("-ignore_error_asterisk", action="store_true")
    parser.add_argument("-debug", action="store_true")
    return parser.parse_args()

def main():

    # 引数取得
    args = get_args()
    print(args)
    targets = ["Tg", "FFV", "Tc", "Density", "Rg"]

    smiles_col = "SMILES_trimmer_cyclic" if args.is_trimmer_cyclic else "SMILES"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    save_model_dir = abspath / "outputs" / args.exp_model / "model"
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # train 読み込み
    train_path = abspath / "outputs" / args.exp_train / "train.csv"
    train = pd.read_csv(train_path)
    if args.debug:
        dfs = []
        for target in targets:
            tmp = train.loc[train[target].notnull()].head(100)
            dfs.append(tmp)
        train = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
        args.n = 1

    if args.ignore_error_asterisk:
        if args.is_trimmer_cyclic:
            cond = train[smiles_col].str.count("\*") == 0
        else:
            cond = train[smiles_col].str.count("\*") == 2
        train = train.loc[cond].reset_index(drop=True)

    print(train.columns)
    print("train shape: ", train.shape)

    # model 読み込み
    cv_model_paths = list(glob(str(abspath / "outputs" / args.exp_model / "model_cv/*.pt")))

    train_config = {
        "task_type": "regression",
        "num_task": 1,
        "batch_size": args.batch_size,
        "augmented_feature": args.augmented_feature,
        "drop_ratio": args.drop_ratio,
        "patience": 50,
        "scheduler_patience": 5,
        "verbose": False,
        "graph_pooling": args.graph_pooling,
    }

    # target 毎に学習
    for target in targets:
        print("=" * 32)
        print(target)

        # model 毎の学習回数を取り出す 
        paths = [path for path in cv_model_paths if target in path]
        fitting_iters = []
        for path in paths:
            model = get_model(args.model)()
            model.load(path)
            fitting_iters.append(len(model.fitting_loss))

        num_epochs = int(np.mean(fitting_iters))
        print(f"num_epochs: {num_epochs}")

        # target が null のデータを除外 
        data = train.loc[train[target].notnull()]
        print("data shape: ", data.shape)
        X = data[smiles_col].to_list()
        y = data[target].to_numpy()

        train_config["epochs"] = num_epochs 

        for seed in range(args.n):
            set_seed(seed)
            model = get_model(args.model)(**train_config)
            model.fit(
                X_train=X,
                y_train=y,
            )
            if args.debug:
                continue
            save_model_path = save_model_dir / f"{args.model}_{seed}_{target}.pt" 
            model.save(path=str(save_model_path))

    if not args.debug:
        dataset_title = f"model-{args.exp_model}"
        dataset_id = f"koya346/{dataset_title}"
        
        create_kaggle_dataset_metadata(dataset_title, dataset_id, save_model_dir)
        upload_kaggle_dataset(dataset_id, save_model_dir)


if __name__ == "__main__":
    main()

