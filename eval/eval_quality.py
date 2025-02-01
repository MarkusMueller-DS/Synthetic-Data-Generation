import pandas as pd
import numpy as np
import json
import os
import argparse
import sys

from dython.nominal import compute_associations
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.metadata import Metadata

parser = argparse.ArgumentParser(description="Evaluation plots")

parser.add_argument(
    "--model", type=str, required=True, help="Name of the model to use (e.g., 'ctgan')"
)

parser.add_argument(
    "--dataset", type=str, required=True, help="Name of the dataset (e.g., 'adult')"
)

parser.add_argument("--ci", type=str, default=None, help="Class imbalance ratio")

args = parser.parse_args()

print(args.model)
print(args.dataset)

DATASET = args.dataset
SDG = args.model
CI = args.ci

# load info json
with open(f"data/info/{DATASET}.json", "r") as f:
    info = json.load(f)

minority_class = info["minority_class"]
target = info["target_col"]

# only compare with minortiy class since it is the relevant one
# if SDG in ["ctgan", "smote", "tvae-all", "tvae-top-2"]:
#    real_path = f"data/processed/{DATASET}/train_min.csv"
# elif SDG in ["tabsyn", "ctab-gan-plus", "vae-bgm"]:
#    real_path = f"data/processed/{DATASET}/train_balanced.csv"

real_path = f"data/processed/{DATASET}/train_min.csv"


if CI != None:
    syn_path = f"data/synthetic/{DATASET}-{CI}/{SDG}.csv"
else:
    syn_path = f"data/synthetic/{DATASET}/{SDG}.csv"

df_real = pd.read_csv(real_path)
df_syn = pd.read_csv(syn_path)

# only use minority class from synthetic data
if SDG in ["tabsyn", "ctab-gan-plus", "vae-bgm"]:
    df_syn = df_syn[df_syn[target] == minority_class]

print(df_syn[target].value_counts())


if DATASET == "adult":
    cat_cols = [
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
elif DATASET == "yeast":
    cat_cols = ["localization.site"]
elif DATASET in ["cc-fraud"]:
    cat_cols = ["Class"]
else:
    print(f"{DATASET} not implemented")

# NaNs are read as float this breaks the code will transfrom to 'missing' for categoricalo columns
for col in cat_cols:
    df_real[col] = df_real[col].replace(0.0, "missing")
    df_syn[col] = df_syn[col].replace(0.0, "missing")


if (SDG == "vae-bgm") and (DATASET == "adult"):
    print("remove column")
    # remove education.num (was removed from paper author)
    df_real.drop(columns=["education.num"], inplace=True)


# generte metadata for sdv evalution metrics
metadata = Metadata.detect_from_dataframe(data=df_real, table_name=DATASET)

print("columns shoudl be the same:", set(df_real.columns) == set(df_syn.columns))

# create diagnostic and quality report from sdv
diagnostic_report = run_diagnostic(
    real_data=df_real, synthetic_data=df_syn, metadata=metadata, verbose=True
)
quality_report = evaluate_quality(
    real_data=df_real, synthetic_data=df_syn, metadata=metadata, verbose=True
)

# Save scores
data_validity_score = diagnostic_report.get_details(property_name="Data Validity")[
    "Score"
].mean()
data_structure_score = diagnostic_report.get_details(property_name="Data Structure")[
    "Score"
].mean()
column_shape_score = quality_report.get_details(property_name="Column Shapes")[
    "Score"
].mean()
column_pair_trends_score = quality_report.get_details(
    property_name="Column Pair Trends"
)["Score"].mean()


# for statistical similarity there sould be no categories which are not in the minority train data
# code wont run when categories are not the same
if DATASET == "adult":
    if SDG == "tabsyn":
        df_syn = df_syn[df_syn["workclass"] != "Never-worked"]
        df_syn = df_syn[df_syn["native.country"] != "Outlying-US(Guam-USVI-etc)"]
    if SDG == "ctab-gan-plus":
        df_syn = df_syn[df_syn["workclass"] != "Without-pay"]
        df_syn = df_syn[df_syn["education"] != "Preschool"]
        df_syn = df_syn[df_syn["native.country"] != "Outlying-US(Guam-USVI-etc)"]
    if SDG == "vae-bgm":
        df_syn = df_syn[df_syn["workclass"] != "Without-pay"]
        df_syn = df_syn[df_syn["workclass"] != "Never-worked"]
        df_syn = df_syn[df_syn["education"] != "Preschool"]
    if SDG == "tvae-all":
        df_syn = df_syn[df_syn["education"] != "Preschool"]


### Statistical similarity
# code from ctab-gan-plus repo
really = df_real.copy()
fakey = df_syn.copy()

# create corrleation matrix
real_corr = compute_associations(df_real, nominal_columns=cat_cols)
syn_corr = compute_associations(df_syn, nominal_columns=cat_cols)

corr_dist = np.linalg.norm(real_corr - syn_corr)

Stat_dict = {}
cat_stat = []
num_stat = []

for column in df_real.columns:

    if column in cat_cols:
        # print(column)

        real_pdf = really[column].value_counts() / really[column].value_counts().sum()
        fake_pdf = fakey[column].value_counts() / fakey[column].value_counts().sum()
        categories = (
            (fakey[column].value_counts() / fakey[column].value_counts().sum())
            .keys()
            .tolist()
        )
        sorted_categories = sorted(categories)

        real_pdf_values = []
        fake_pdf_values = []

        for i in sorted_categories:
            real_pdf_values.append(real_pdf[i])
            fake_pdf_values.append(fake_pdf[i])

        if len(real_pdf) != len(fake_pdf):
            zero_cats = set(really[column].value_counts().keys()) - set(
                fakey[column].value_counts().keys()
            )
            for z in zero_cats:
                real_pdf_values.append(real_pdf[z])
                fake_pdf_values.append(0)
        Stat_dict[column] = distance.jensenshannon(
            real_pdf_values, fake_pdf_values, 2.0
        )
        cat_stat.append(Stat_dict[column])
        # print("column: ", column, "JSD: ", Stat_dict[column])
    else:
        scaler = MinMaxScaler()
        scaler.fit(df_real[column].values.reshape(-1, 1))
        l1 = scaler.transform(df_real[column].values.reshape(-1, 1)).flatten()
        l2 = scaler.transform(df_syn[column].values.reshape(-1, 1)).flatten()
        Stat_dict[column] = wasserstein_distance(l1, l2)
        # print("column: ", column, "WD: ", Stat_dict[column])
        num_stat.append(Stat_dict[column])

print(np.mean(num_stat), np.mean(cat_stat), corr_dist)

# check if there is already a csv to save the results
if not os.path.exists("results/quality_data.csv"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    init_data = [
        {
            "Timestamp": ts,
            "Syn_Algo": "init",
            "Dataset": "init",
            "Validity Score": 0,
            "Structure Score": 0,
            "Column Shape Quality": 0,
            "Column Pair Trends": 0,
            "Average WD (Continuous Columns)": 0,
            "Average JSD (Categorical Columns)": 0,
            "Correlation Distance": 0,
        }
    ]
    resutls_df = pd.DataFrame(init_data)
    resutls_df.to_csv("results/quality_data.csv", index=False)

df_old = pd.read_csv("results/quality_data.csv")

# add ci label when dataset is cc-fraud
dataset = DATASET
if dataset == "cc-fraud":
    dataset += f"-{CI}"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
new_data = [
    {
        "Timestamp": ts,
        "Syn_Algo": SDG,
        "Dataset": dataset,
        "Validity Score": data_validity_score,
        "Structure Score": data_structure_score,
        "Column Shape Quality": column_shape_score,
        "Column Pair Trends": column_pair_trends_score,
        "Average WD (Continuous Columns)": np.mean(num_stat),
        "Average JSD (Categorical Columns)": np.mean(cat_stat),
        "Correlation Distance": corr_dist,
    }
]

df_new = pd.DataFrame(new_data)

df_new = pd.concat([df_old, df_new])
df_new.to_csv("results/quality_data.csv", index=False)

print(df_new)
