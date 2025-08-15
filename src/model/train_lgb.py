import time
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

import lightgbm as lgb

import numpy as np


def save_lgb_model(model, output_path: str):
    model.save_model(output_path)

def load_lgb_model(model_path: str):
    return lgb.Booster(model_file=model_path)

def train_lgb_for_target(train, target_col, features, params, output_dir: Path, n_splits=5):
    loss_tables = []
    print(f"\n=== Training for target: {target_col} ===")

    df_train = train[~train[target_col].isna()]

    X = df_train[features]
    y = df_train[target_col]
    
    oof = np.zeros(len(X))
    models = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)



    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        loss_table = {}
        print(f"fold: {fold + 1}")
        X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(200)
            ]
        )

        save_lgb_model(model, str(output_dir / f"model_{target_col}_{fold}.txt"))

        pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof[val_idx] = pred
        mse = mean_squared_error(y_val, pred)
        mae = mean_absolute_error(y_val, pred)
        loss_table["fold"] = fold
        loss_table["target"] = target_col
        loss_table["mae"] = mae
        loss_table["mse"] = mse

        loss_tables.append(loss_table)
        models.append(model)

    score_mse = mean_squared_error(y, oof)
    score_mae = mean_absolute_error(y, oof)
    print(f"RMSE for {target_col}: {score_mse:.4f}")
    print(f"MAE for {target_col}: {score_mae:.4f}")
    
    return models, oof, df_train["id"].values, loss_tables

