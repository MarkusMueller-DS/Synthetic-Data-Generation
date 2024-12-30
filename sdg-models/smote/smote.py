import os
import argparse
import json

import pandas as pd
from imblearn.over_sampling import SMOTENC, SMOTE

parser = argparse.ArgumentParser(
    description="Args for synthetic data generation with SMOTE"
)

parser.add_argument(
    "--dataset", type=str, required=True, help="Name of the dataset (e.g., 'adult')"
)
DATA_PATH = f"../../data"
INFO_PATH = f"../../data/info/"


# need to use src dataset
# target column does not need to be transformed or mentioned in the cat cols
def smote_train_gen(dataset):
    print(f"train and generate data for {dataset}")

    # load info file
    # load info json
    with open(f"{INFO_PATH}/{dataset}.json", "r") as f:
        info = json.load(f)

    target = info["target_col"]

    train_path = f"{DATA_PATH}/processed/{dataset}/train_src.csv"

    # define categorial columns if ther are any
    if dataset in ["adult"]:
        cat_features = info["cat_col_names"]
    else:
        cat_features = None

    train_df = pd.read_csv(train_path)

    # print(train_df)

    X = train_df.iloc[:, :-1]
    y = train_df[target]

    # Use SMOTENC when mixed columns types
    # Use SMOTE for only numerical column types
    if dataset in ["adult"]:
        smote_nc = SMOTENC(categorical_features=cat_features, random_state=42)
        X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    elif dataset in ["yeast"]:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

    print("SMOTE finished")

    # Combine X_resampled and y_resampled into a single DataFrame
    data_resampled = X_resampled.copy()
    data_resampled[target] = y_resampled

    # Indenify synthetic rows
    original_indices = X.index
    resampled_indices = X_resampled.index
    synthetic_indices = resampled_indices.difference(original_indices)

    # Filter synthetic rows
    synthetic_samples = data_resampled.loc[synthetic_indices]

    # save syn data
    save_path = f"{DATA_PATH}/synthetic/{dataset}/smote.csv"
    synthetic_samples.to_csv(save_path, index=False)
    print("saved synthtic data here:", save_path)
    print("Finished data generation")


if __name__ == "__main__":
    args = parser.parse_args()
    DATASET = args.dataset
    if DATASET in ["adult", "yeast"]:
        smote_train_gen(DATASET)
    else:
        print(f"{DATASET} not implemented")
