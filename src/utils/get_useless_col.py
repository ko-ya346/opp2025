import numpy as np

def get_useless_cols(df, threshold_null=0.9, threshold_corr=0.95):
    feature_cols = [col for col in df.columns]

    # ユニーク数が1の特徴量
    unique_1_cols = [col for col in feature_cols if df[col].nunique() == 1]
    print("Unique=1 col:", unique_1_cols)

    # 欠損が多い特徴量
    almost_null_cols = [col for col in feature_cols if (df[col].isnull().sum() / len(df)) >= threshold_null]

    valid_cols = [col for col in feature_cols if col not in unique_1_cols + almost_null_cols]

    # 相関係数が高い列は片方取り除く
    df_corr = df[valid_cols].corr().abs()
    upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper.columns if any(upper[col] >= threshold_corr)]

    print("Highly correlated cols: ", high_corr_cols)
    return list(set(unique_1_cols + almost_null_cols + high_corr_cols))
