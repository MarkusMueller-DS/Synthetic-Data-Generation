import sys
import os
import json
import pandas as pd
import numpy as np
import argparse
import shutil
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="process dataset")

parser.add_argument("--dataset", type=str, default=None, help="Name of dataset")
args = parser.parse_args()

INFO_PATH = "data/info"


def process_yeast():
    # no missing values
    # only numerical columns
    # remove first column from dataset
    # split data

    # create folder structure
    os.makedirs("data/processed/yeast", exist_ok=True)
    os.makedirs("data/synthetic/yeast", exist_ok=True)

    # read info json
    with open(f"{INFO_PATH}/yeast.json", "r") as f:
        info = json.load(f)

    # load relevatn information from info json
    data_path = info["data_path"]
    majority_class = info["majority_class"]
    minority_class = info["minority_class"]
    target = info["target_col"]
    column_names = info["column_names"]
    header = info["header"]

    # multiple spaces as speeration
    df = pd.read_csv(data_path, sep="\s+", header=header)

    # add column names
    df.columns = column_names

    # remove unrelevant column
    df.drop(columns=["Sequence.Name"], inplace=True)

    # filter for minortiy and majoirty class
    df = df[(df[target] == majority_class) | (df[target] == minority_class)]

    # create train and test splits
    X = df.iloc[:, :-1]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42, test_size=0.2
    )

    # combine X and y for train and test
    X_train[target] = y_train
    X_test[target] = y_test

    df_train = X_train
    df_test = X_test

    # create different splite of training data
    train_min = df_train[df_train[target] == minority_class]
    train_maj_sampled = df_train[X_train[target] == majority_class].sample(
        n=train_min.shape[0], random_state=42
    )
    train_balanced = pd.concat([train_min, train_maj_sampled])
    # shuffle train_balanced
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(
        drop=True
    )

    # save processed datasets
    save_path = "data/processed/yeast"
    train_min.to_csv(f"{save_path}/train_min.csv", index=False)
    train_balanced.to_csv(f"{save_path}/train_balanced.csv", index=False)
    df_train.to_csv(f"{save_path}/train_src.csv", index=False)
    df_test.to_csv(f"{save_path}/test.csv", index=False)

    print("finished yeast processing")


def process_data(dataset):
    if dataset == "yeast":
        print("Process yeast dataset")
        process_yeast()


if __name__ == "__main__":
    process_data(args.dataset)
