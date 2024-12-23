import pandas as pd
import numpy as np
import json
import os
import argparse

from dython.nominal import compute_associations
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# from synthcity.metrics import eval_detection, eval_performance, eval_statistical
# from synthcity.plugins.core.dataloader import GenericDataLoader
# from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser(description="Evaluation plots")

parser.add_argument(
    "--model", type=str, required=True, help="Name of the model to use (e.g., 'ctgan')"
)

parser.add_argument(
    "--dataset", type=str, required=True, help="Name of the dataset (e.g., 'adult')"
)

args = parser.parse_args()

print(args.model)
print(args.dataset)


# load info json
with open("sdg-models/tabsyn/data/info/adult.json", "r") as f:
    info = json.load(f)

DATASET = args.dataset
SDG = args.model

if SDG == "tabsyn":
    real_path = "sdg-models/tabsyn/synthetic/adult/real.csv"
    syn_path = "sdg-models/tabsyn/synthetic/adult/tabsyn.csv"
elif SDG == "ctab-gan-plus":
    real_path = "sdg-models/ctab-gan-plus/Real_Datasets/adult/train.csv"
    syn_path = "sdg-models/ctab-gan-plus/Fake_Datasets/adult/adult_fake_0.csv"
else:
    print(f"{SDG} not implemented")


df_real = pd.read_csv(real_path)
df_syn = pd.read_csv(syn_path)

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
else:
    print(f"{DATASET} not implemented")

# NaNs are read as float this breaks the code will transfrom to 'missing' for categoricalo columns
for col in cat_cols:
    df_real[col] = df_real[col].replace(0.0, "missing")
    df_syn[col] = df_syn[col].replace(0.0, "missing")


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
            "Average WD (Continuous Columns)": 0,
            "Average JSD (Categorical Columns)": 0,
            "Correlation Distance": 0,
        }
    ]
    resutls_df = pd.DataFrame(init_data)
    resutls_df.to_csv("results/quality_data.csv", index=False)

df_old = pd.read_csv("results/quality_data.csv")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
new_data = [
    {
        "Timestamp": ts,
        "Syn_Algo": SDG,
        "Dataset": DATASET,
        "Average WD (Continuous Columns)": np.mean(num_stat),
        "Average JSD (Categorical Columns)": np.mean(cat_stat),
        "Correlation Distance": corr_dist,
    }
]

df_new = pd.DataFrame(new_data)

df_new = pd.concat([df_old, df_new])
df_new.to_csv("results/quality_data.csv", index=False)

print(df_new)


### alpha-Precision & beta-Recall
# Segmentation fault ???
# Code from TabSyn
def alpha_beta():
    real_data = df_real.copy()
    syn_data = df_syn.copy()

    # both dataframes need to be the same shape
    if syn_data.shape[0] > real_data.shape[0]:
        syn_data = syn_data.sample(real_data.shape[0])

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]
    cat_col_idx += target_col_idx

    num_real_data = real_data[num_col_idx]
    cat_real_data = real_data[cat_col_idx]

    num_real_data_np = num_real_data.to_numpy()
    cat_real_data_np = cat_real_data.to_numpy().astype("str")

    num_syn_data = syn_data[num_col_idx]
    cat_syn_data = syn_data[cat_col_idx]

    num_syn_data_np = num_syn_data.to_numpy()

    # cat_syn_data_np = np.array
    cat_syn_data_np = cat_syn_data.to_numpy().astype("str")

    encoder = OneHotEncoder()
    encoder.fit(cat_real_data_np)

    cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
    cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()

    le_real_data = pd.DataFrame(
        np.concatenate((num_real_data_np, cat_real_data_oh), axis=1)
    ).astype(float)
    le_real_num = pd.DataFrame(num_real_data_np).astype(float)
    le_real_cat = pd.DataFrame(cat_real_data_oh).astype(float)

    le_syn_data = pd.DataFrame(
        np.concatenate((num_syn_data_np, cat_syn_data_oh), axis=1)
    ).astype(float)
    le_syn_num = pd.DataFrame(num_syn_data_np).astype(float)
    le_syn_cat = pd.DataFrame(cat_syn_data_oh).astype(float)

    np.set_printoptions(precision=4)

    result = []

    print("=========== All Features ===========")
    print("Data shape: ", le_syn_data.shape)

    X_syn_loader = GenericDataLoader(le_syn_data)
    X_real_loader = GenericDataLoader(le_real_data)

    quality_evaluator = eval_statistical.AlphaPrecision()
    qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
    qual_res = {
        k: v for (k, v) in qual_res.items() if "naive" in k
    }  # use the naive implementation of AlphaPrecision
    qual_score = np.mean(list(qual_res.values()))

    print(
        "alpha precision: {:.6f}, beta recall: {:.6f}".format(
            qual_res["delta_precision_alpha_naive"],
            qual_res["delta_coverage_beta_naive"],
        )
    )

    Alpha_Precision_all = qual_res["delta_precision_alpha_naive"]
    Beta_Recall_all = qual_res["delta_coverage_beta_naive"]
