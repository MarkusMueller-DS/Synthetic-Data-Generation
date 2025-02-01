import pandas as pd
import os
import argparse
import sys
import pickle
import json

from sdv.metadata import Metadata
from sdv.evaluation.single_table import get_column_plot
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

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


# config
DATASET = args.dataset
SDG = args.model
CI = args.ci
# only use minority class
SCALER_FLAG = "min"

if CI != None:
    output_folder_cat = f"results/plots/{DATASET}-{CI}/{SDG}/categorial"
    output_folder_num = f"results/plots/{DATASET}-{CI}/{SDG}/numerical"
    output_folder_tsne = f"results/plots/{DATASET}-{CI}/{SDG}/tsne"
else:
    output_folder_cat = f"results/plots/{DATASET}/{SDG}/categorial"
    output_folder_num = f"results/plots/{DATASET}/{SDG}/numerical"
    output_folder_tsne = f"results/plots/{DATASET}/{SDG}/tsne"

os.makedirs(output_folder_cat, exist_ok=True)
os.makedirs(output_folder_num, exist_ok=True)
os.makedirs(output_folder_tsne, exist_ok=True)


# load info file
info_path = f"data/info/{DATASET}.json"
with open(info_path, "r") as f:
    info = json.load(f)

target = info["target_col"]
minority_class = info["minority_class"]

# onyl compare minority class
# if SDG in ["ctgan", "smote", "tvae-all", "tvae-top-2"]:
#    real_path = f"data/processed/{DATASET}/train_min.csv"
#    SCALER_FLAG = "min"
# elif SDG in ["ctab-gan-plus", "tabsyn", "vae-bgm"]:
#    real_path = f"data/processed/{DATASET}/train_balanced.csv"
#    SCALER_FLAG = "balanced"
# else:
#    print(f"{SDG} not implemented")
#    sys.exit(1)

if SDG in [
    "ctgan",
    "smote",
    "tvae-all",
    "tvae-top-2",
    "ctab-gan-plus",
    "tabsyn",
    "vae-bgm",
]:
    real_path = f"data/processed/{DATASET}/train_min.csv"
else:
    print(f"{SDG} not implemented")
    sys.exit(1)


def impute_missing_values(df, info):
    # Category: repalce nan with "missing"
    cat_columns = info["cat_col_names"]
    num_columns = info["num_col_names"]

    for col in cat_columns:
        df[col] = df[col].fillna("Missing")
    for col in num_columns:
        df[col] = df[col].fillna(0.0)

    return df


if CI != None:
    syn_path = f"data/synthetic/{DATASET}-{CI}/{SDG}.csv"
else:
    syn_path = f"data/synthetic/{DATASET}/{SDG}.csv"

df_real = pd.read_csv(real_path)
df_syn = pd.read_csv(syn_path)

# filter synthetic data to only containt the minority class label
df_syn = df_syn[df_syn[target] == minority_class]
assert df_syn[target].unique() == minority_class

# load in the source train dataset to generate the labels for the labelEncoder
if CI != None:
    real_src_path = f"data/processed/{DATASET}/train_src_{CI}.csv"
else:
    real_src_path = f"data/processed/{DATASET}/train_src.csv"
df_real_src = pd.read_csv(real_src_path)

metadata = Metadata.detect_from_dataframe(df_real)

if DATASET == "adult":
    # configure column names
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
    # vae-bgm model drops education.num column
    if SDG == "vae-bgm":
        num_cols = [
            "age",
            "fnlwgt",
            "capital.gain",
            "capital.loss",
            "hours.per.week",
        ]
    else:
        num_cols = [
            "age",
            "fnlwgt",
            "education.num",
            "capital.gain",
            "capital.loss",
            "hours.per.week",
        ]


elif DATASET == "yeast":
    cat_cols = ["localization.site"]
    num_cols = ["mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc"]
elif DATASET == "cc-fraud":
    cat_cols = ["Class"]
    num_cols = [
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount",
    ]
else:
    print(f"{DATASET} not implemented")
    sys.exit(1)


def column_plots():
    # generate plots for categorical columns
    for col in cat_cols:
        fig_cat = get_column_plot(
            real_data=df_real, synthetic_data=df_syn, metadata=metadata, column_name=col
        )
        fig_cat.write_image(f"{output_folder_cat}/{col}.png")
        # fig_cat.show()

    # generate plots for numerical columns
    for col in num_cols:
        fig_num = get_column_plot(
            real_data=df_real, synthetic_data=df_syn, metadata=metadata, column_name=col
        )
        fig_num.write_image(f"{output_folder_num}/{col}.png")
        # fig_num.show()


####
# T-sne
####
def tsne():
    global df_syn

    # there are to many synthetic instances for cc-fraud
    # ther is no meaning behind the plot just a could of points
    if DATASET == "cc-fraud":
        if CI == "1":
            n = int(len(df_syn) / 2)
            print(n)
            df_syn = df_syn.sample(n=n, random_state=42)
        # if CI == "5":
        #    n = int(len(df_syn) / 2)
        #    print(n)
        #    df_syn = df_syn.sample(n=n, random_state=42)

    # Initialize variables
    tsne_model = TSNE(n_components=2, random_state=42)
    scaler = StandardScaler()

    # check if tsne from train data exists if not create tsne
    # only using minority class
    # if SDG in ["ctgan", "smote", "tvae-all", "tvae-top-2"]:
    #    real_path_tsne = f"data/processed/{DATASET}/train_min_tsne.csv"
    # elif SDG in ["ctab-gan-plus", "tabsyn", "vae-bgm"]:
    #    real_path_tsne = f"data/processed/{DATASET}/train_balanced_tsne.csv"
    real_path_tsne = f"data/processed/{DATASET}/train_min_tsne.csv"

    # label encoder is universal since it is trained on the the soruce dataset
    if not os.path.exists(f"data/processed/{DATASET}/label_encoder.pkl"):
        print("Generate LabelEncoder from src train dataset")
        label_encoders = {}
        for col in df_real_src.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            le.fit(df_real_src[col])
            label_encoders[col] = le
        # save label_endocers
        with open(f"data/processed/{DATASET}/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoders, f)

    # outdated: this is not universal there will be two scalers on for dataset with only minority class and for dataset with a downsampled majority class to reach a balanced dataset
    if not os.path.exists(real_path_tsne):
        # load label encoders
        with open(f"data/processed/{DATASET}/label_encoder.pkl", "rb") as f:
            label_encoders = pickle.load(f)

        # encode columns
        for col in df_real_src.select_dtypes(include=["object", "category"]).columns:
            df_real[col] = label_encoders[col].transform(df_real[col])

        print("Generate train tsne")
        # tsne from train dataset
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(df_real)

        if SCALER_FLAG == "min" and not os.path.exists(
            f"data/processed/{DATASET}/scaler_min.pkl"
        ):
            # save scaler min
            with open(f"data/processed/{DATASET}/scaler_min.pkl", "wb") as f:
                pickle.dump(scaler, f)
        # not relevant anymore
        # if SCALER_FLAG == "balanced" and not os.path.exists(
        #    f"data/processed/{DATASET}/scaler_balanced.pkl"
        # ):
        #    # save scaler balanced
        #    with open(f"data/processed/{DATASET}/scaler_balanced.pkl", "wb") as f:
        #        pickle.dump(scaler, f)

        # generate tsne
        tsne_result = tsne_model.fit_transform(normalized_data)

        # save dataset
        tsne_train_df = pd.DataFrame(tsne_result, columns=["Dim1", "Dim2"])
        tsne_train_df.to_csv(real_path_tsne, index=False)

    # load label encoders
    with open(f"data/processed/{DATASET}/label_encoder.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    if SCALER_FLAG == "min":
        scaler_path = f"data/processed/{DATASET}/scaler_min.pkl"
    # outdated
    # else:
    #    scaler_path = f"data/processed/{DATASET}/scaler_balanced.pkl"

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # load tsne train dataset
    print("load train tsne")
    tsne_train_df = pd.read_csv(real_path_tsne)

    for col in df_syn.select_dtypes(include=["object", "category"]).columns:
        df_syn[col] = label_encoders[col].transform(df_syn[col])

    # check for missing values:
    if df_syn.isnull().values.any():
        print(df_syn.isnull().sum())
        df_syn = impute_missing_values(df_syn, info)

    # normalize data
    # add column with average value since it is not part of the vae-bgm model
    if DATASET == "adult" and SDG == "vae-bgm":
        df_syn["education.num"] = 12.0
        # Spalten neu anordnen
        df_syn = df_syn[scaler.feature_names_in_]
    normalized_data = scaler.transform(df_syn)

    print("Generate test tsne")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_syn = tsne.fit_transform(normalized_data)

    tsne_syn_df = pd.DataFrame(tsne_syn, columns=["Dim1", "Dim2"])

    # combine datasets
    tsne_train_df["Label"] = "Real"
    tsne_syn_df["Label"] = "Generated"

    tsne_df = pd.concat([tsne_train_df, tsne_syn_df])

    ## adjust axis for each dataset
    # yeast ctgan       (-20 20), (-25 20)
    # yeast smote       (-30 35), (-15 15)
    # yeast tvae-all    (-18 22), (-18 20)
    # yeast tvae-top-2  (-30 30), (-15 14)
    # yeast tabsyn      (-18 22), (-22 25)
    # yeast vae-bgm     (-20 15), (-20 15)
    # yeast             (-30 35), (-25 25)

    # adult ctgan           (-110 100), (-110 100)
    # adult smote           (-100 115), (-110 110)
    # adult tvae-all        (-110 100), (-110 100)
    # adult tvae-top-2      (-120 110), (-110 120)
    # adult tabsyn          (-110 100), (-100 100)
    # adult ctab-gan-plus   (-110 100), (-100 100)
    # adult vae-bgm         (-120 110), (-100 110)
    # adult                 (-120 115), (-110 120)

    # cc-fraud-1 ctgan          (-70 80), (-80 80)
    # cc-fraud-1 smote          (-140 140), (-125 125)
    # cc-fraud-1 tvae-all       (-100 100), (-100 100)
    # cc-fraud-1 tvae-top-2     (-80 90), (-80 80)
    # cc-fraud-1 tabsyn         (-80 100), (-90 90)
    # cc-fraud-1 ctag-gan-plus  (-100 100), (-60 60)
    # cc-fraud-1 vae-bgm        (-110 100), (-110 100)
    # cc-fraud-1                (-125 125), (-120 120)

    # cc-fraud-5 ctgan          (-60 60), (-65 65)
    # cc-fraud-5 smote          (-100 100), (-100 90)
    # cc-fraud-5 tvae-all       (-70 100), (-70 70)
    # cc-fraud-5 tvae-top-2     (-60 70), (-65 65)
    # cc-fraud-5 tabsyn         (-80 80), (-80 80)
    # cc-fraud-5 ctag-gan-plus  (-80 80), (-45 45)
    # cc-fraud-5 vae-bgm        (-110 100), (-65 60)
    # cc-fraud-5                (-110 100), (-100 90)

    # x_min, x_max = -110, 100
    # y_min, y_max = -100, 90

    plt.figure(figsize=(8, 6))

    generated_subset = tsne_df[tsne_df["Label"] == "Generated"]
    plt.scatter(
        generated_subset["Dim1"],
        generated_subset["Dim2"],
        label="Generated",
        alpha=0.5,
        s=10,
        color="orange",
    )
    real_subset = tsne_df[tsne_df["Label"] == "Real"]
    plt.scatter(
        real_subset["Dim1"],
        real_subset["Dim2"],
        label="Real",
        alpha=0.5,
        s=10,
        color="blue",
    )
    # set axis limits
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)

    plt.title(f"t-SNE {DATASET} - {SDG}")
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f"{output_folder_tsne}/{SDG}.png", dpi=300, bbox_inches="tight")


column_plots()
tsne()

print("Finished plots creation")
print(f"Distribution of categorical columns: {output_folder_cat}")
print(f"Distribution of numerical columns: {output_folder_num}")
print(f"TSNE: {output_folder_tsne}")
