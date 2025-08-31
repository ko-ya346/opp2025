from pathlib import Path
from glob import glob

import json
import os
import sys
import argparse
import pandas as pd
import numpy as np

import lightgbm as lgb


abspath = Path("/home/kouya-takahashi/kaggle/opp2025")
sys.path.append(str(abspath))

from src.model import save_lgb_model
from src.utils.upload_kaggle_dataset import (
    create_kaggle_dataset_metadata,
    upload_kaggle_dataset,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_train", type=str, help="学習データの実験ナンバー")
    parser.add_argument("exp_model", type=str, help="モデルの実験ナンバー")
    parser.add_argument("-n", type=int, default=5, help="異なる seed で学習する回数")
    return parser.parse_args()

def main():
    # 引数取得
    args = get_args()
    print(args)
    targets = ["Tg", "FFV", "Tc", "Density", "Rg"]

    train_path = abspath / "outputs" / args.exp_train / "train.csv"
    cv_model_paths = list(glob(str(abspath / "outputs" / args.exp_model / "model_cv/*.txt")))
    save_model_dir = abspath / "outputs" / args.exp_model / "model"
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # 保存していた best_iterations を呼び出す 
    # {target: [best_iter1, , best_iter2, ..]}
    best_iterations_path = abspath / "outputs" / args.exp_train / "best_iterations.json"
    with open(best_iterations_path, "r") as f:
        best_iterations = json.load(f)
    
    # 学習データ読み込み
    train = pd.read_csv(train_path)
    print(train.shape)

    # 学習時のパラメータ取得
    # TODO: fold 毎に異なるパラメータを取得する場合があるので修正対応
    org_params = lgb.Booster(model_file=cv_model_paths[0]).params
    # print(org_params)

    for target in targets:
        print(target)
        # model を 1つ呼び出す
        features = lgb.Booster(model_file=[path for path in cv_model_paths if target in path][0]).feature_name()
        print("feature num: ", len(features))

        # cv の best_iteration の平均を計算
        mean_best_iteration = int(np.mean(best_iterations[target]))
        print("mean best iteration: ", mean_best_iteration)

        # target が null のデータを除外 
        data = train.loc[train[target].notnull()]
        print("data shape: ", data.shape)
        X = data[features]
        y = data[target]

        dtrain = lgb.Dataset(X, label=y)

        # seed を変えて学習
        for seed in range(args.n):
            params = org_params.copy()
            params["seed"] = seed
            params["num_iterations"] = mean_best_iteration

            model = lgb.train(
                params,
                dtrain
            )

            # 学習モデルを保存する
            filename = f"model_{target}_{seed}.txt"
            print(f"Save model: {filename}")
            save_lgb_model(model, str(save_model_dir / filename))

    dataset_title = f"model-{args.exp_model}"
    dataset_id = f"koya346/{dataset_title}"
    
    create_kaggle_dataset_metadata(dataset_title, dataset_id, save_model_dir)
    upload_kaggle_dataset(dataset_id, save_model_dir)


if __name__ == "__main__":
    main()

