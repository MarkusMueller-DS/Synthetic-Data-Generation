import os
import json
import pandas as pd
import numpy as np
import argparse
import shutil


parser = argparse.ArgumentParser(description="process dataset")

parser.add_argument("--dataset", type=str, default=None, help="Name of dataset")
args = parser.parse_args()


def process_adult():
    print("Processing adult dataset")

    print("Check if info.json is present in folder")
    INFO_PATH = "data/info/adult.json"
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
    train_df = pd.read_csv(
        info["data_path"], header=info["header"], skipinitialspace=True
    )

    # get columns
    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    column_names = info["column_names"]
    target_col_idx = info["target_col_idx"]

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]

    # load test, which needs more processing
    if info["test_path"]:
        # if testing data is given
        test_path = info["test_path"]

        with open(test_path, "r") as f:
            lines = f.readlines()[1:]
            test_save_path = f"data/raw/adult/test.data"
            if not os.path.exists(test_save_path):
                with open(test_save_path, "a") as f1:
                    for line in lines:
                        save_line = line.strip("\n").strip(".")
                        f1.write(f"{save_line}\n")

        test_df = pd.read_csv(test_save_path, header=None, skipinitialspace=True)

    # ToDo: do I really need this?
    # add more information to info json
    col_info = {}
    for col_idx in num_col_idx:
        col_info[col_idx] = {
            "type": "numerical",
            "max": float(train_df[col_idx].max()),
            "min": float(train_df[col_idx].min()),
        }

    for col_idx in cat_col_idx:
        col_info[col_idx] = {
            "type": "categorical",
            "categories": list(set(train_df[col_idx])),
        }

    # target col info
    col_info[target_col_idx] = {
        "type": "categorical",
        "categories": list(set(train_df[col_idx])),
    }

    info["column_info"] = col_info

    print(info["column_info"])

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

    # Todo: add validation and information about train & test

    # Save data specifc for tabsyn
    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df["income"].to_numpy()

    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()
    y_test = test_df["income"].to_numpy()

    np.save(f"{PROCESSED_DATA_PATH}/X_num_train.npy", X_num_train)
    np.save(f"{PROCESSED_DATA_PATH}/X_cat_train.npy", X_cat_train)
    np.save(f"{PROCESSED_DATA_PATH}/y_train.npy", y_train)

    np.save(f"{PROCESSED_DATA_PATH}/X_num_test.npy", X_num_test)
    np.save(f"{PROCESSED_DATA_PATH}/X_cat_test.npy", X_cat_test)
    np.save(f"{PROCESSED_DATA_PATH}/y_test.npy", y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)

    train_df.to_csv(f"{PROCESSED_DATA_PATH}/train.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DATA_PATH}/test.csv", index=False)

    # add more information to info.json

    # add paths to info.json
    info["train_path_processed"] = f"{PROCESSED_DATA_PATH}/train.csv"
    info["test_path_processed"] = f"{PROCESSED_DATA_PATH}/test.csv"

    info["column_names"] = column_names
    info["train_num"] = train_df.shape[0]
    info["test_num"] = test_df.shape[0]

    print("Numerical", X_num_train.shape)
    print("Categorical", X_cat_train.shape)

    metadata = {"columns": {}}

    for i in num_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "numerical"
        metadata["columns"][i]["computer_representation"] = "Float"

    for i in cat_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "categorical"

    metadata["columns"][target_col_idx] = {}
    metadata["columns"][target_col_idx]["sdtype"] = "categorical"

    info["metadata"] = metadata

    with open(f"{PROCESSED_DATA_PATH}/info.json", "w") as file:
        json.dump(info, file, indent=4)


def process_data(name_dataset):
    if name_dataset == "adult":
        # can be removed once testing is done
        if os.path.exists("data/processed/adult"):
            shutil.rmtree("data/processed/adult")
            print("deleting exising files")
        process_adult()


if __name__ == "__main__":
    process_data(args.dataset)
