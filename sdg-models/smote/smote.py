import os
import argparse

import pandas as pd
from imblearn.over_sampling import SMOTE

parser = argparse.ArgumentParser(
    description="Args for synthetic data generation with SMOTE"
)

parser.add_argument(
    "--dataset", type=str, required=True, help="Name of the dataset (e.g., 'adult')"
)


args = parser.parse_args()


DATASET = args.dataset

print(args.dataset)


## Preprocess
# -> numerical format

if DATASET == "adult":
    pass
