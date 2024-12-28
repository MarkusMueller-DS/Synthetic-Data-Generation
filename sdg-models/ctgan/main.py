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
DATA_PATH = "../../data"


def ctgan_train_gen(dataset):
    print(f"train and generate data for {dataset} dataset")

    # read info json
    with open(f"{DATA_PATH}/info/{dataset}.json", "r") as f:
        info = json.load(f)

    minority_class = info["minority_class"]
    majority_class = info["majority_class"]
    target = info["target_col"]

    # load data
    df_train = pd.read_csv(f"{DATA_PATH}/processed/{DATASET}/train_min.csv")

    df_train[target] = df_train[target].map({minority_class: 1})

    # create specific metadata object of ctgan
    metadata = Metadata.detect_from_dataframe(data=df_train, table_name=dataset)
    # print(metadata)

    # create synthetsizer
    synthesizer = CTGANSynthesizer(metadata)

    # fit to train data
    print("Start training")
    synthesizer.fit(df_train)

    # generate synthetic data
    print("Generate synthetic samples")
    if dataset == "yeast":
        n = 329
    elif dataset == "adult":
        n = 16879
    syn_data = synthesizer.sample(num_rows=n)

    # transform target column back to string values
    syn_data[target] = syn_data[target].map({1: minority_class})

    # save syn data
    save_path = f"{DATA_PATH}/synthetic/{DATASET}/ctgan.csv"
    syn_data.to_csv(save_path, index=False)
    print("saved synthtic data here:", save_path)
    print("Finished data generation")


if __name__ == "__main__":
    if DATASET in ["yeast", "adult"]:
        ctgan_train_gen(DATASET)
    else:
        print(f"{DATASET} not implemented")
