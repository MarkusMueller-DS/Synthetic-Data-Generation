import os
import sys
import argparse
import json

import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata


parser = argparse.ArgumentParser(
    description="Args for synthetic data generation with SMOTE"
)

parser.add_argument(
    "--dataset", type=str, required=True, help="Name of the dataset (e.g., 'adult')"
)

args = parser.parse_args()

DATASET = args.dataset
INFO_PATH = f"data/info/{DATASET}.json"


# load info file
# load info json
with open(INFO_PATH, "r") as f:
    info = json.load(f)


def get_min_maj(df, info):
    minority_class = info["minortiy_class"]
    majority_class = info["majority_class"]

    count_min = df[df["income"] == minority_class].shape[0]
    count_max = df[df["income"] == majority_class].shape[0]

    count_sample = count_max - count_min

    print(count_max, count_min)
    print(count_sample)

    return count_sample


print("train")

# load dataset
if DATASET == "adult":
    train_path = "data/processed/adult/train.csv"
else:
    print(f"{DATASET} not implemented")
    sys.exit(1)


train_df = pd.read_csv(train_path)

print(train_df)

# find minority and majoirity class
minority_class = info["minority_class"]

# filter Dataframe
df_train_min = train_df[train_df["income"] == minority_class]
print(df_train_min)

# CTGAN needs specific metadata
metadata = Metadata.detect_from_dataframe(data=df_train_min, table_name=DATASET)
print(metadata)

# create and train CTGAN
# add cuda argument
# synthesizer = CTGANSynthesizer(metadata, cuda="cuda:1", verbose=True)
synthesizer = CTGANSynthesizer(metadata, verbose=True, epochs=1)
print("start training")
synthesizer.fit(df_train_min)

# print(synthesizer.get_loss_values())

print("sample")
num_sample = get_min_maj(train_df, info)

# sample
print("num_sample:", num_sample)
syn_data = synthesizer.sample(num_rows=num_sample)

print(syn_data["income"].value_counts())

syn_data.to_csv("data/synthetic/adult/syn_data.csv", index=False)
