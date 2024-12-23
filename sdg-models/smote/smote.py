import os
import argparse
import json

import pandas as pd
from imblearn.over_sampling import SMOTENC

parser = argparse.ArgumentParser(
    description="Args for synthetic data generation with SMOTE"
)

parser.add_argument(
    "--dataset", type=str, required=True, help="Name of the dataset (e.g., 'adult')"
)


args = parser.parse_args()


DATASET = args.dataset
INFO_PATH = f"data/info/{DATASET}.json"

print(args.dataset)

# load info file
# load info json
with open(INFO_PATH, "r") as f:
    info = json.load(f)

if DATASET == "adult":
    train_path = "data/processed/adult/train.csv"
    cat_features = [
        "workclass",
        "education",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native.country",
    ]
else:
    print("Dataset not implemented")

train_df = pd.read_csv(train_path)

# print(train_df)

X = train_df.iloc[:, :-1]
y = train_df["income"]

smote_nc = SMOTENC(categorical_features=cat_features, random_state=42)
X_resampled, y_resampled = smote_nc.fit_resample(X, y)
print("SMOTE finished")


# combine X and y
X["income"] = y

X.to_csv("data/synthetic/final_data.csv", index=False)
