from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

import lightgbm as lgb

import numpy as np


def save_lgb_model(model, output_path: str):
    model.save_model(output_path)

def load_lgb_model(model_path: str):
    return lgb.Booster(model_file=model_path)

def train_lgb_for_target(train, test, target_col, features, params, output_dir: Path, n_splits=5):
    print(f"\n=== Training for target: {target_col} ===")

    df_train = train[~train[target_col].isna()]
    df_test = test.copy()

    X = df_train[features]
    y = df_train[target_col]
    X_test = df_test[features]
    
    preds_test = np.zeros(len(X_test))
    oof = np.zeros(len(X))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
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

        save_lgb_model(model, str(output_dir / f"model_{fold}.txt"))

        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        preds_test += model.predict(X_test, num_iteration=model.best_iteration) / n_splits



    score_mse = mean_squared_error(y, oof)
    score_mae = mean_absolute_error(y, oof)
    print(f"RMSE for {target_col}: {score_mse:.4f}")
    print(f"MAE for {target_col}: {score_mae:.4f}")
    
    return preds_test, oof, df_train["id"].values

