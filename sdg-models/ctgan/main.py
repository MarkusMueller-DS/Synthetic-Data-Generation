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

    # create specific metadata object of ctgan
    metadata = Metadata.detect_from_dataframe(data=df_train, table_name=dataset)
    # print(metadata)

    # create synthetsizer
    synthesizer = CTGANSynthesizer(metadata, verbose=True)

    # fit to train data
    print("Start training")
    synthesizer.fit(df_train)

    # generate synthetic data
    print("Generate synthetic samples")
    if dataset == "yeast":
        n = 329
    if dataset == "adult":
        n = 16879
    if dataset == "cc-fraud":
        n_1 = 38612
        n_5 = 7092

    if dataset == "cc-fraud":
        syn_data_1 = synthesizer.sample(num_rows=n_1)
        syn_data_5 = synthesizer.sample(num_rows=n_5)

        # save syn data
        save_path_1 = f"{DATA_PATH}/synthetic/cc-fraud-1/ctgan.csv"
        save_path_5 = f"{DATA_PATH}/synthetic/cc-fraud-5/ctgan.csv"

        os.makedirs(f"{DATA_PATH}/synthetic/cc-fraud-1", exist_ok=True)
        os.makedirs(f"{DATA_PATH}/synthetic/cc-fraud-5", exist_ok=True)

        syn_data_1.to_csv(save_path_1, index=False)
        syn_data_5.to_csv(save_path_5, index=False)
        print("saved synthtic data here:", save_path_1)
        print("saved synthtic data here:", save_path_5)
        print("Finished data generation")
    else:
        syn_data = synthesizer.sample(num_rows=n)

        # save syn data
        save_path = f"{DATA_PATH}/synthetic/{dataset}/ctgan.csv"
        syn_data.to_csv(save_path, index=False)
        print("saved synthtic data here:", save_path)
        print("Finished data generation")


if __name__ == "__main__":
    if DATASET in ["yeast", "adult", "cc-fraud"]:
        ctgan_train_gen(DATASET)
    else:
        print(f"{DATASET} not implemented")
