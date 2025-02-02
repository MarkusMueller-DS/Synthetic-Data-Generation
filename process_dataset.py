import sys
import os
import json
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="process dataset")

parser.add_argument("--dataset", type=str, default=None, help="Name of dataset")
args = parser.parse_args()

INFO_PATH = "data/info"


def process_yeast(dataset):
    # no missing values
    # only numerical columns
    # remove first column from dataset
    # split data

    # create folder structure
    os.makedirs(f"data/processed/{dataset}", exist_ok=True)
    os.makedirs(f"data/synthetic/{dataset}", exist_ok=True)

    # read info json
    with open(f"{INFO_PATH}/{dataset}.json", "r") as f:
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

    # drop the first column
    df.drop(columns=[0], inplace=True)

    # add column names
    df.columns = column_names

    # remove unrelevant column
    # df.drop(columns=["Sequence.Name"], inplace=True)

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
    save_path = f"data/processed/{dataset}"
    train_min.to_csv(f"{save_path}/train_min.csv", index=False)
    train_balanced.to_csv(f"{save_path}/train_balanced.csv", index=False)
    df_train.to_csv(f"{save_path}/train_src.csv", index=False)
    df_test.to_csv(f"{save_path}/test.csv", index=False)

    print("finished yeast processing")


def process_adult(dataset):
    # create folder structure
    os.makedirs(f"data/processed/{dataset}", exist_ok=True)
    os.makedirs(f"data/synthetic/{dataset}", exist_ok=True)

    # read info json
    with open(f"{INFO_PATH}/{dataset}.json", "r") as f:
        info = json.load(f)

    # load relevatn information from info json
    data_path = info["data_path"]
    test_path = info["test_path"]
    majority_class = info["majority_class"]
    minority_class = info["minority_class"]
    target = info["target_col"]
    column_names = info["column_names"]
    header = info["header"]
    num_columns = info["num_col_names"]
    cat_columns = info["cat_col_names"]

    # read data
    train_df = pd.read_csv(data_path, header=header, skipinitialspace=True)

    with open(test_path, "r") as f:
        lines = f.readlines()[1:]
        test_save_path = f"data/raw/{dataset}/test.data"
        if not os.path.exists(test_save_path):
            with open(test_save_path, "a") as f1:
                for line in lines:
                    save_line = line.strip("\n").strip(".")
                    f1.write(f"{save_line}\n")

    test_df = pd.read_csv(test_save_path, header=header, skipinitialspace=True)

    # add column names
    train_df.columns = column_names
    test_df.columns = column_names

    # transfrom missing values to standard format
    # missing values are marked with ?
    for col in num_columns:
        train_df.loc[train_df[col] == "?", col] = np.nan
        test_df.loc[test_df[col] == "?", col] = np.nan

    for col in cat_columns:
        train_df.loc[train_df[col] == "?", col] = "nan"
        test_df.loc[test_df[col] == "?", col] = "nan"

    # create different splite of training data
    train_min = train_df[train_df[target] == minority_class]
    train_maj_sampled = train_df[train_df[target] == majority_class].sample(
        n=train_min.shape[0], random_state=42
    )
    train_balanced = pd.concat([train_min, train_maj_sampled])
    # shuffle train_balanced
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(
        drop=True
    )

    # save processed datasets
    save_path = f"data/processed/{dataset}"
    train_min.to_csv(f"{save_path}/train_min.csv", index=False)
    train_balanced.to_csv(f"{save_path}/train_balanced.csv", index=False)
    train_df.to_csv(f"{save_path}/train_src.csv", index=False)
    test_df.to_csv(f"{save_path}/test.csv", index=False)

    print(f"finished {dataset} processing")


def process_ccfraud(dataset):
    # no missing values
    # remove time column

    # test and train balanced is the same between the different class imblanace ratios
    # create folder structure
    os.makedirs(f"data/processed/cc-fraud")

    # 1% Class imbalance
    one_ci = "-1"
    os.makedirs(f"data/synthetic/{dataset + one_ci}", exist_ok=True)
    # 5% class imbalance
    five_ci = "-5"
    os.makedirs(f"data/synthetic/{dataset + five_ci}", exist_ok=True)

    # read info json
    with open(f"{INFO_PATH}/{dataset}-1.json", "r") as f:
        info = json.load(f)

    # load relevatn information from info json
    data_path = info["data_path"]
    majority_class = info["majority_class"]
    minority_class = info["minority_class"]
    target = info["target_col"]

    data_df = pd.read_csv(data_path)
    # drop time column
    data_df.drop(columns=["Time"], inplace=True)

    # transform target column into sting values
    # SOTA models expect that the target column is a string value
    data_df["Class"] = data_df["Class"].map({0: majority_class, 1: minority_class})
    # print(data_df)

    # split data
    train_data, test_data = train_test_split(
        data_df, test_size=0.2, stratify=data_df["Class"]
    )

    print(train_data[target].value_counts())

    # create different class imbalance ratios
    def create_imbalance_dataset(train_data, minority_ratio):
        fraud = train_data[train_data[target] == minority_class]
        non_fraud = train_data[train_data[target] == majority_class]

        # Calculate the number of majority sampled to keep
        majority_size = int(len(fraud) / minority_ratio) - len(fraud)

        # Undersample the majority class
        non_fraud_sampled = non_fraud.sample(n=majority_size, random_state=42)

        # Combine and shuffle
        imbalanced_data = pd.concat([fraud, non_fraud_sampled]).sample(
            frac=1, random_state=42
        )
        return imbalanced_data

    train_df_1 = create_imbalance_dataset(train_data, 0.01)
    train_df_5 = create_imbalance_dataset(train_data, 0.05)

    # create different splite of training data for the new class imbalance ratio datasets
    # 1% class imbalance ratio
    train_min = data_df[data_df[target] == minority_class]
    train_maj_sampled = data_df[data_df[target] == majority_class].sample(
        n=train_min.shape[0], random_state=42
    )
    # combien min and maj sampeld to create balanced dataset
    train_balanced = pd.concat([train_min, train_maj_sampled])
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(
        drop=True
    )

    # save processed datasets
    save_path = f"data/processed/{dataset}"

    # train_min is the same between cc-fraud-1 and cc-fraud-5
    train_min.to_csv(f"{save_path}/train_min.csv", index=False)
    # train_balanced is the same between cc-fraud-1 and cc-fraud-5
    train_balanced.to_csv(f"{save_path}/train_balanced.csv", index=False)
    # Class Imbalance datast is different
    train_df_1.to_csv(f"{save_path}/train_src_1.csv", index=False)
    train_df_5.to_csv(f"{save_path}/train_src_5.csv", index=False)
    # test is the same
    test_data.to_csv(f"{save_path}/test.csv", index=False)

    print(f"finished {dataset} processing")


if __name__ == "__main__":
    DATASET = args.dataset
    if DATASET == "yeast":
        print("Process yeast dataset")
        process_yeast(DATASET)
    elif DATASET == "adult":
        print("Process adult dataset")
        process_adult(DATASET)
    elif DATASET == "cc-fraud":
        print("Process creditcard fraud dataset")
        process_ccfraud(DATASET)
    else:
        print(f"{DATASET} not implemented")
        sys.exit(1)
