import numpy as np
import pandas as pd
import os
import sys
import json
import argparse
import shutil
from sklearn.preprocessing import LabelEncoder

name = "adult"

INFO_PATH = "Real_Datasets/info"

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
    test_save_path = f"Real_Datasets/{name}/test.data"
    if not os.path.exists(test_save_path):
        with open(test_save_path, "a") as f1:
            for line in lines:
                save_line = line.strip("\n").strip(".")
                f1.write(f"{save_line}\n")
test_df = pd.read_csv(test_save_path, header=None, skipinitialspace=True)

majority_class = info["majority_class"]
minority_class = info["minority_class"]
target_col = info["target_col_idx"]

minority_df = train_df[train_df[target_col] == minority_class]
majoirty_df = train_df[train_df[target_col] == majority_class]

minority_size = len(minority_df)
majoirty_sampled = majoirty_df.sample(n=minority_size, random_state=42)
train_df = pd.concat([minority_df, majoirty_sampled]).sample(frac=1, random_state=42)

print(train_df.shape)
print(train_df[target_col].value_counts())

cat_col_idx_list = info["cat_col_idx"]

column_names = info["column_names"]
# add column names to df
train_df.columns = column_names
test_df.columns = column_names

print(train_df)
print(test_df)

train_df.to_csv("Real_Datasets/adult/train.csv", index=False)
test_df.to_csv("Real_Datasets/adult/test.csv", index=False)
