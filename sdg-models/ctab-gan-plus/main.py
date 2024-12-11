import torch
from model.ctabgan import CTABGAN
from model.eval.evaluation import get_utility_metrics, stat_sim, privacy_metrics
import numpy as np
import pandas as pd
import glob


num_exp = 1
dataset = "Adult"
real_path = "Real_Datasets/test.csv"
fake_file_root = "Fake_Datasets"

synthesizer = CTABGAN(
    raw_csv_path=real_path,
    test_ratio=0.20,
    categorical_columns=[
        "workclass",
        "education",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native.country",
        "income",
    ],
    log_columns=[],
    mixed_columns={"capital.loss": [0.0], "capital.gain": [0.0]},
    general_columns=["age"],
    non_categorical_columns=[],
    integer_columns=["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week", "education.num"],
    problem_type={"Classification": "income"},
)


for i in range(num_exp):
    synthesizer.fit()
    syn = synthesizer.generate_samples()
    syn.to_csv(
        fake_file_root
        + "/"
        + dataset
        + "/"
        + dataset
        + "_fake_{exp}.csv".format(exp=i),
        index=False,
    )
