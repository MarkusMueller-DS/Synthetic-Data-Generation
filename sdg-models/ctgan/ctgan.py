import os
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from datetime import datetime


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
    SAVE_PATH = f'models/adult/ctgan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    synthesizer.save(SAVE_PATH)


def sample(dataset):
    print("sample")
    pass


def main(args):
    print("Gude from ctgan")

    dataset = args.dataset
    if args.mode == "train":
        train(dataset)
    if args.mode == "sample":
        sample(dataset)
