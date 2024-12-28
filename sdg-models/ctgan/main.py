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


def ctgan_yeast():
    print("train and generate data for yeast dataset")

    # read info json
    with open(f"{DATA_PATH}/info/yeast.json", "r") as f:
        info = json.load(f)

    # load data
    df_train = pd.read_csv(f"{DATA_PATH}/processed/{DATASET}/train_min.csv")

    df_train["localization.site"] = df_train["localization.site"].map(
        {"CYT": 0, "ME2": 1}
    )

    # create specific metadata object of ctgan
    metadata = Metadata.detect_from_dataframe(data=df_train, table_name="yeast")
    # print(metadata)

    # create synthetsizer
    synthesizer = CTGANSynthesizer(metadata)

    # fit to train data
    print("Start training")
    synthesizer.fit(df_train)

    # generate synthetic data
    print("Generate synthetic samples")
    syn_data = synthesizer.sample(num_rows=329)

    # transform target column back to string values
    syn_data["localization.site"] = syn_data["localization.site"].map({1: "ME2"})

    # save syn data
    save_path = f"{DATA_PATH}/synthetic/{DATASET}/ctgan.csv"
    syn_data.to_csv(save_path, index=False)
    print("saved synthtic data here:", save_path)
    print("Finished data generation")


if __name__ == "__main__":
    if DATASET == "yeast":
        ctgan_yeast()
