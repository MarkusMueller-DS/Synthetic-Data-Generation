import torch
from model.ctabgan import CTABGAN
from model.eval.evaluation import get_utility_metrics, stat_sim, privacy_metrics
import numpy as np
import pandas as pd
import glob
import argparse
import json

parser = argparse.ArgumentParser(
    description="Args for synthetic data generation with ctag-gan-plus"
)

parser.add_argument(
    "--dataset", type=str, required=True, help="Name of the dataset (e.g., 'adult')"
)

DATA_PATH = "../../data"
INFO_PATH = "../../data/info"


def train_gen(dataset):

    # load info json
    with open(f"{INFO_PATH}/{dataset}.json", "r") as f:
        info = json.load(f)

    target = info["target_col"]

    train_path = f"{DATA_PATH}/processed/{dataset}/train_balanced.csv"

    if dataset == "adult":
        categorical_columns = [
            "workclass",
            "education",
            "marital.status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native.country",
            "income",
        ]
        log_columns = []
        mixed_columns = {"capital.loss": [0.0], "capital.gain": [0.0]}
        general_columns = ["age"]
        non_categorical_columns = []
        integer_columns = [
            "age",
            "fnlwgt",
            "capital.gain",
            "capital.loss",
            "hours.per.week",
            "education.num",
        ]
        # define the amount of data to be generated
        n = 33758
    elif dataset == "yeast":
        categorical_columns = ["localization.site"]
        log_columns = []
        mixed_columns = {}
        general_columns = []
        non_categorical_columns = []
        integer_columns = ["mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc"]
        # define the amount of data to be generated
        n = 329
    else:
        print(f"{dataset} not implemented")

    synthesizer = CTABGAN(
        raw_csv_path=train_path,
        test_ratio=0.20,
        categorical_columns=categorical_columns,
        log_columns=log_columns,
        mixed_columns=mixed_columns,
        general_columns=general_columns,
        non_categorical_columns=non_categorical_columns,
        integer_columns=integer_columns,
        problem_type={"Classification": target},
    )

    synthesizer.fit()
    syn_data = synthesizer.generate_samples(n)

    # save syn data
    save_path = f"{DATA_PATH}/synthetic/{dataset}/ctab-gan-plus.csv"
    syn_data.to_csv(save_path, index=False)
    print("saved synthtic data here:", save_path)
    print("Finished data generation")


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset
    if dataset in ["adult", "yeast"]:
        train_gen(dataset)
    else:
        print(f"{dataset} not implemented")
