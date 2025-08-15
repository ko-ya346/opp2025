from sklearn.mixture import GaussianMixture
import pandas as pd

def add_augumented_gmm(df, y, n_samples=1000, n_components=5, random_state=42):
    df["target"] = y

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(df)
    synthetic_data, _ = gmm.sample(n_samples) 
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
    aug_df = pd.concat([df, synthetic_df]).reset_index(drop=True)
    X_augumented = aug_df.drop("target", axis=1)
    y_augumented = aug_df["target"]
    return X_augumented, y_augumented
