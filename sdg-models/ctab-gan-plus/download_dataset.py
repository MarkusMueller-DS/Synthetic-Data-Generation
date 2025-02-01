import os
import zipfile
from urllib import request
import shutil


DATA_DIR = "Real_Datasets/"

NAME_URL_DICT_UCI = {"adult": "https://archive.ics.uci.edu/static/public/2/adult.zip"}


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
    for name in NAME_URL_DICT_UCI.keys():
        delete_files(name)
        download_from_uci(name)
