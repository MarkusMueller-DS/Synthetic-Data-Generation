import numpy as np
import pandas as pd
import os
import json
import argparse

# from sklearn.preprocessing import LabelEncoder


parser = argparse.ArgumentParser(description="Processing data for SMOTE")

parser.add_argument(
    "--dataset", type=str, required=True, help="Name of the dataset (e.g., 'adult')"
)

args = parser.parse_args()

name = args.dataset

INFO_PATH = "data/info"

# read info json
with open(f"{INFO_PATH}/{name}.json", "r") as f:
    info = json.load(f)

# print(info)
data_path = info["data_path"]
if info["file_type"] == "csv":
    train_df = pd.read_csv(data_path, header=info["header"], skipinitialspace=True)

test_path = info["test_path"]
# read test
with open(test_path, "r") as f:
    lines = f.readlines()[1:]
    test_save_path = f"data/raw/{name}/test.data"
    if not os.path.exists(test_save_path):
        with open(test_save_path, "a") as f1:
            for line in lines:
                save_line = line.strip("\n").strip(".")
                f1.write(f"{save_line}\n")
test_df = pd.read_csv(test_save_path, header=None, skipinitialspace=True)

column_names = info["column_names"]
# add column names to df
train_df.columns = column_names
test_df.columns = column_names


print(train_df)
print(test_df)

train_df.to_csv(f"data/processed/{name}/train.csv", index=False)
test_df.to_csv(f"data/processed/{name}/test.csv", index=False)


print("Finished processing data for CTGAN")
