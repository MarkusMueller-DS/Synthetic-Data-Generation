import os
import argparse
import json

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

parser = argparse.ArgumentParser(description="Args for random over and under sampling")

parser.add_argument(
    "--dataset", type=str, required=True, help="Name of the dataset (e.g., 'adult')"
)
parser.add_argument(
    "--ci", type=str, default=None, help="Name of the dataset (e.g., 'adult')"
)

args = parser.parse_args()
DATASET = args.dataset
CI = args.ci

DATA_PATH = "../../data"
INFO_PATH = "../../data/info/"


def rus(dataset):
    print("starting random udersampling process")
    with open(f"{INFO_PATH}/{dataset}.json", "r") as f:
        info = json.load(f)

    target_col = info["target_col"]

    if CI != None:
        train_path = f"{DATA_PATH}/processed/{dataset}/train_src_{CI}.csv"
    else:
        train_path = f"{DATA_PATH}/processed/{dataset}/train_src.csv"
    train_df = pd.read_csv(train_path)

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    rus = RandomUnderSampler(random_state=42)
    X_train_resample, y_train_resample = rus.fit_resample(X, y)

    # combine X and y
    X_train_resample[target_col] = y_train_resample

    df_final = X_train_resample
    print(df_final[target_col].value_counts())

    # save data
    # save syn data
    if CI != None:
        save_path = f"{DATA_PATH}/synthetic/{dataset}-{CI}/rus.csv"
        # create folder if not exists
        os.makedirs(f"{DATA_PATH}/synthetic/{dataset}-{CI}", exist_ok=True)
    else:
        save_path = f"{DATA_PATH}/synthetic/{dataset}/rus.csv"
        os.makedirs(f"{DATA_PATH}/synthetic/{dataset}", exist_ok=True)
    df_final.to_csv(save_path, index=False)
    print("saved data here:", save_path)
    print("Finished random undersmapling")


def ros(dataset):
    print("starting random oversampling process")
    with open(f"{INFO_PATH}/{dataset}.json", "r") as f:
        info = json.load(f)

    target_col = info["target_col"]

    if CI != None:
        train_path = f"{DATA_PATH}/processed/{dataset}/train_src_{CI}.csv"
    else:
        train_path = f"{DATA_PATH}/processed/{dataset}/train_src.csv"
    train_df = pd.read_csv(train_path)

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    rus = RandomOverSampler(random_state=42)
    X_train_resample, y_train_resample = rus.fit_resample(X, y)

    # combine X and y
    X_train_resample[target_col] = y_train_resample

    df_final = X_train_resample
    print(df_final[target_col].value_counts())

    # save data
    if CI != None:
        save_path = f"{DATA_PATH}/synthetic/{dataset}-{CI}/ros.csv"
        # create folder if not exists
        os.makedirs(f"{DATA_PATH}/synthetic/{dataset}-{CI}", exist_ok=True)
    else:
        save_path = f"{DATA_PATH}/synthetic/{dataset}/ros.csv"
        os.makedirs(f"{DATA_PATH}/synthetic/{dataset}", exist_ok=True)
    df_final.to_csv(save_path, index=False)
    print("saved data here:", save_path)
    print("Finished random oversmapling")


if __name__ == "__main__":
    if DATASET in ["adult", "yeast", "cc-fraud"]:
        rus(DATASET)
        ros(DATASET)
    else:
        print(f"{DATASET} not implemented")
