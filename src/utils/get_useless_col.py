import numpy as np

def get_useless_cols(df, threshold_corr=0.95):
    feature_cols = [col for col in df.columns]


    # ユニーク数が1の特徴量
    unique_1_cols = [col for col in feature_cols if df[col].nunique() == 1]
    print("Unique=1 col:", unique_1_cols)

    valid_cols = [col for col in feature_cols if col not in unique_1_cols]
    df_corr = df[valid_cols].corr().abs()
    upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper.columns if any(upper[col] >= threshold_corr)]

    print("Highly correlated cols: ", high_corr_cols)
    return unique_1_cols + high_corr_cols
