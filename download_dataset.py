import os
import zipfile
from urllib import request
import shutil
import argparse


DATA_DIR = "data/raw"

NAME_URL_DICT_UCI = {
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "yeast": "https://archive.ics.uci.edu/static/public/110/yeast.zip",
}

parser = argparse.ArgumentParser(description="download dataset")

# General configs
parser.add_argument("--dataset", type=str, default=None, help="Name of dataset.")
args = parser.parse_args()

print(args.dataset)


# Test if data exists and if ther is also processed
# deletet everything for faster testing
# can be removed once testing is done
def delete_files(name):
    if os.path.exists(f"{DATA_DIR}/{name}"):
        shutil.rmtree(f"{DATA_DIR}/{name}")
        print("deleted existing folder")


def unzip_file(zip_filepath, dest_path):
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(dest_path)


def download_from_uci(name):
    print(f"Start downloading dataset {name} from UCI")
    save_dir = f"{DATA_DIR}/{name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("created folder")

        url = NAME_URL_DICT_UCI[name]
        request.urlretrieve(url, f"{save_dir}/{name}.zip")
        print(
            f"Finish downloading dataset from {url}, data has been saved to {save_dir}."
        )

        unzip_file(f"{save_dir}/{name}.zip", save_dir)
        print(f"Finish unzipping {name}.")

    else:
        print("Already downloaded")


if __name__ == "__main__":
    if args.dataset in ["yeast", "adult"]:
        delete_files(args.dataset)
        download_from_uci(args.dataset)
    if args.dataset == "cc-fraud":
        print(
            "please download dataset from here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
        print("save datast here data/raw/cc-fraud/creaditcard.csv")
        os.makedirs("data/raw/cc-fraud", exist_ok=True)
