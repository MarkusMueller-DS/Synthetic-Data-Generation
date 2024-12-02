import os
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata


# ToDo: add Cuda implementation
# https://github.com/sdv-dev/SDV/blob/main/sdv/single_table/ctgan.py
# runs fine on local cpu but would be faster with gpu


def get_min_maj(dataset):
    DATA_PATH = f"data/processed/{dataset}/train.csv"
    train_df = pd.read_csv(DATA_PATH)

    counts = train_df["income"].value_counts()
    minority_category = counts.idxmin()
    majority_category = counts.idxmax()

    count_min = train_df[train_df["income"] == minority_category].shape[0]
    count_max = train_df[train_df["income"] == majority_category].shape[0]

    count_sample = count_max - count_min

    print(count_max, count_min)
    print(count_sample)

    return count_sample


def train(args):
    print("train")

    # load dataset
    DATA_PATH = f"data/processed/{args.dataset}/train.csv"

    train_df = pd.read_csv(DATA_PATH)

    print(train_df)

    # ToDo: refactor -> information is now in info json
    # find minority and majoirity class
    counts = train_df["income"].value_counts()
    minority_category = counts.idxmin()

    # filter Dataframe
    df_train_min = train_df[train_df["income"] == minority_category]
    print(df_train_min)

    # CTGAN needs specific metadata
    metadata = Metadata.detect_from_dataframe(data=df_train_min, table_name=args.model)

    # create and train CTGAN
    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(df_train_min)

    # print(synthesizer.get_loss_values())

    # Save synthesizer
    os.makedirs(f"models/{args.dataset}", exist_ok=True)
    save_path = f"models/{args.dataset}/{args.model}.pkl"
    synthesizer.save(save_path)

    print(f"Synthesizer saved in: {save_path}")


def sample(args):
    print("sample")
    num_sample = get_min_maj(args.dataset)

    # load the synthesizer
    synthesizer = CTGANSynthesizer.load(
        filepath=f"models/{args.dataset}/{args.model}.pkl"
    )

    # sample
    syn_data = synthesizer.sample(num_rows=num_sample)

    print(syn_data)

    os.makedirs(f"data/synthetic/{args.dataset}", exist_ok=True)
    save_path = f"data/synthetic/{args.dataset}/{args.model}.csv"
    syn_data.to_csv(save_path, index=False)

    print(f"Synthetic data saved in:{save_path}")


def main(args):
    print("Gude from ctgan")

    if args.mode == "train":
        train(args)
    if args.mode == "sample":
        sample(args)
