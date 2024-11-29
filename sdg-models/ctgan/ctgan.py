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


def train(dataset):
    print("train")

    # load dataset
    DATA_PATH = f"data/processed/{dataset}/train.csv"

    train_df = pd.read_csv(DATA_PATH)

    print(train_df)

    # find minority and majoirity class
    counts = train_df["income"].value_counts()
    minority_category = counts.idxmin()

    # filter Dataframe
    df_train_min = train_df[train_df["income"] == minority_category]
    print(df_train_min)

    # CTGAN needs specific metadata
    metadata = Metadata.detect_from_dataframe(data=df_train_min, table_name="adult")

    # create and train CTGAN
    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(df_train_min)

    print(synthesizer.get_loss_values())

    # Save synthesizer
    os.makedirs("models/adult", exist_ok=True)
    save_path = "models/adult/ctgan.pkl"
    synthesizer.save(save_path)

    print(f"Synthesizer saved in: {save_path}")


def sample(dataset):
    print("sample")
    num_sample = get_min_maj(dataset)

    # load the synthesizer
    synthesizer = CTGANSynthesizer.load(filepath=f"models/{dataset}/ctgan.pkl")

    # sample
    syn_data = synthesizer.sample(num_rows=num_sample)

    print(syn_data)

    os.makedirs(f"data/synthetic/{dataset}", exist_ok=True)
    save_path = f"data/synthetic/{dataset}/ctgan.csv"
    syn_data.to_csv(save_path, index=False)

    print(f"Synthetic data saved in:{save_path}")


def main(args):
    print("Gude from ctgan")

    dataset = args.dataset
    if args.mode == "train":
        train(dataset)
    if args.mode == "sample":
        sample(dataset)
