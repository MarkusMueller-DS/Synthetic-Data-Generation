import pandas as pd
import os
import argparse

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

args = parser.parse_args()

print(args.model)
print(args.dataset)


# config
DATASET = args.dataset
SDG = args.model

output_folder_cat = f"results/plots/{DATASET}/{SDG}/categorial"
output_folder_num = f"results/plots/{DATASET}/{SDG}/numerical"
output_folder_tsne = f"results/plots/{DATASET}/{SDG}/tsne"

os.makedirs(output_folder_cat, exist_ok=True)
os.makedirs(output_folder_num, exist_ok=True)
os.makedirs(output_folder_tsne, exist_ok=True)


if SDG == "tabsyn":
    real_path = "sdg-models/tabsyn/synthetic/adult/real.csv"
    syn_path = "sdg-models/tabsyn/synthetic/adult/tabsyn.csv"

if SDG == "ctag-gan-plus":
    real_path = f"sdg-models/ctag-gan-plus/Real_Datasets/{DATASET}/train.csv"
    syn_path = f"sdg-models/ctag-gan-plus/Fake_Datasets/{DATASET}/train.csv"


df_real = pd.read_csv(real_path)
df_syn = pd.read_csv(syn_path)

# load info json
# with open("../sdg-models/tabsyn/data/info/adult.json", "r") as f:
#    info = json.load(f)

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
    num_cols = [
        "age",
        "fnlwgt",
        "education.num",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
    ]

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
    fig_cat.write_image(f"{output_folder_num}/{col}.png")
    # fig_num.show()

####
# T-sne
####

print("creating Tsne")

if df_syn.shape[0] > df_real.shape[0]:
    df_sample = df_syn.sample(df_real.shape[0])

    # combine dfs
    combined_df = pd.concat([df_real, df_syn], ignore_index=True)
    labels = ["Real"] * len(df_real) + ["Generated"] * len(df_syn)

    # apply label enoding
    label_encoders = {}
    for col in combined_df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col])
        label_encoders[col] = le

    # normalize data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(combined_df)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(normalized_data)

    tsne_df = pd.DataFrame(tsne_results, columns=["Dim1", "Dim2"])
    tsne_df["Label"] = labels

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

    # Plot "Real" second (foreground)
    real_subset = tsne_df[tsne_df["Label"] == "Real"]
    plt.scatter(
        real_subset["Dim1"],
        real_subset["Dim2"],
        label="Real",
        alpha=0.5,
        s=10,
        color="blue",
    )

    plt.title("t-SNE Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f"{output_folder_tsne}/tsne.png", dpi=300, bbox_inches="tight")


print("Finished plots creation")
print(f"Distribution of categorical columns: {output_folder_cat}")
print(f"Distribution of numerical columns: {output_folder_num}")
print(f"TSNE: {output_folder_tsne}")
