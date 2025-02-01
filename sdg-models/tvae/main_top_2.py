import os
import sys
import argparse
import json

import pandas as pd
from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata

parser = argparse.ArgumentParser(
    description="Args for synthetic data generation with SMOTE"
)

parser.add_argument(
    "--dataset", type=str, required=True, help="Name of the dataset (e.g., 'adult')"
)

parser.add_argument("--ci", type=str, default=None, help="Class imbalance ratio")

args = parser.parse_args()

DATASET = args.dataset
DATA_PATH = "../../data"
CI = args.ci


def tvae_train_gen(dataset):
    print(f"train and generate data for {dataset} dataset")

    # read info json
    with open(f"{DATA_PATH}/info/{dataset}.json", "r") as f:
        info = json.load(f)

    target = info["target_col"]
    minority_class = info["minority_class"]

    if dataset == "cc-fraud":
        dataset += f"-{CI}"
    print(dataset)

    # load synthetic data from to 2 SDG models
    ctgan_path = f"{DATA_PATH}/synthetic/{dataset}/ctgan.csv"
    tabsyn_path = f"{DATA_PATH}/synthetic/{dataset}/tabsyn.csv"
    vae_bgm_path = f"{DATA_PATH}/synthetic/{dataset}/vae-bgm.csv"
    ctab_gan_plus_path = f"{DATA_PATH}/synthetic/{dataset}/ctab-gan-plus.csv"

    ctgan_df = pd.read_csv(ctgan_path)
    tabsyn_df = pd.read_csv(tabsyn_path)
    vae_bgm_df = pd.read_csv(vae_bgm_path)
    if dataset != "yeast":
        ctab_gan_plus_df = pd.read_csv(ctab_gan_plus_path)

    # filter tabsyn, vae-bgm and ctab-gan-plus for minority class
    # both are generated with minority and majoirty datapoints
    tabsyn_df = tabsyn_df[tabsyn_df[target] == minority_class]
    vae_bgm_df = vae_bgm_df[vae_bgm_df[target] == minority_class]
    if dataset != "yeast":
        ctab_gan_plus_df = ctab_gan_plus_df[ctab_gan_plus_df[target] == minority_class]

    # load data for best models
    # adult = TabSyn + CTAB-GAN-Plus
    if dataset == "adult":
        df_1 = tabsyn_df
        df_2 = ctab_gan_plus_df
    # yeast = CTGAN + TabSyn
    if dataset == "yeast":
        df_1 = ctgan_df
        df_2 = tabsyn_df
    # cc-fraud = CTGAN + VAE-BGM
    if dataset in ["cc-fraud-1", "cc-fraud-5"]:
        df_1 = ctgan_df
        df_2 = vae_bgm_df

    # combine datasets and create new train dataset
    # find min rows among the two dataframes so that the distrinution is fair and not biases towards a specifc model
    df_list = [df_1, df_2]
    min_rows = min(len(df) for df in df_list)
    print("min_rows:", min_rows)
    sampled_dfs = [df.sample(n=min_rows, random_state=42) for df in df_list]
    train_df = pd.concat(sampled_dfs, ignore_index=True)

    # save train_df
    if dataset in ["cc-fraud-1", "cc-fraud-5"]:
        train_df.to_csv(
            f"{DATA_PATH}/processed/cc-fraud/tvae_top2_{CI}.csv", index=False
        )
    else:
        train_df.to_csv(f"{DATA_PATH}/processed/{dataset}/tvae_top2.csv", index=False)

    # create metadata object for TVAE
    metadata = Metadata.detect_from_dataframe(data=train_df, table_name=dataset)

    # create synthetsizer
    synthesizer = TVAESynthesizer(metadata, verbose=True)

    # fit to train data
    print("Start training")
    synthesizer.fit(train_df)

    print("Generate synthetic samples")
    if dataset == "yeast":
        n = 329
    if dataset == "adult":
        n = 16879
    if dataset == "cc-fraud-1":
        n = 38612
    if dataset == "cc-fraud-5":
        n = 7029

    # generating new data
    print("n:", n)
    syn_data = synthesizer.sample(num_rows=n)

    # transform target column back to string values
    # syn_data[target] = syn_data[target].map({1: minority_class})

    print(syn_data)

    # save syn data
    save_path = f"{DATA_PATH}/synthetic/{dataset}/tvae-top-2.csv"
    os.makedirs(f"{DATA_PATH}/synthetic/{dataset}", exist_ok=True)
    syn_data.to_csv(save_path, index=False)
    print("saved synthtic data here:", save_path)
    print("Finished data generation")


if __name__ == "__main__":
    if DATASET in ["yeast", "adult", "cc-fraud"]:
        tvae_train_gen(DATASET)
    else:
        print(f"{DATASET} is not implemented")
