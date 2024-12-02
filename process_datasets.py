import os
import json
import pandas as pd
import numpy as np
import argparse


DATA_PATH = "data/raw"

parser = argparse.ArgumentParser(description="process dataset")

parser.add_argument("--dataset", type=str, default=None, help="Name of dataset")
args = parser.parse_args()


def process_adult():
    print("Processing adult dataset")

    print("Check if info.json is present in folder")
    INFO_PATH = f"{DATA_PATH}/adult/adult.json"
    print("info_path:", INFO_PATH)
    if not os.path.exists(INFO_PATH):
        raise FileNotFoundError(f"The file does not exists")
    else:
        print("info.json found")

    # load info json
    with open(INFO_PATH, "r") as f:
        info = json.load(f)
    # print(info)

    # load train dataset
    train_df = pd.read_csv(info["train_path_raw"], header=None, skipinitialspace=True)

    # get columns
    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    column_names = info["column_names"]

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]

    # load test, which needs more processing
    if info["test_path_raw"]:
        # if testing data is given
        test_path = info["test_path_raw"]

        with open(test_path, "r") as f:
            lines = f.readlines()[1:]
            test_save_path = f"data/raw/adult/test.data"
            if not os.path.exists(test_save_path):
                with open(test_save_path, "a") as f1:
                    for line in lines:
                        save_line = line.strip("\n").strip(".")
                        f1.write(f"{save_line}\n")

        test_df = pd.read_csv(test_save_path, header=None, skipinitialspace=True)

    # add column names to dataframes
    train_df.columns = column_names
    test_df.columns = column_names

    # handle missing values
    for col in num_columns:
        print()
        train_df.loc[train_df[col] == "?", col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == "?", col] = "nan"
    for col in num_columns:
        test_df.loc[test_df[col] == "?", col] = np.nan
    for col in cat_columns:
        test_df.loc[test_df[col] == "?", col] = "nan"

    # save data
    PROCESSED_DATA_PATH = f"data/processed/{args.dataset}"
    # certe folder
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    train_df.to_csv(info["train_path_processed"], index=False)
    test_df.to_csv(info["test_path_processed"], index=False)

    # Todo: add validation and information about train & test


def process_data(name_dataset):
    if name_dataset == "adult":
        process_adult()


if __name__ == "__main__":
    process_data(args.dataset)
