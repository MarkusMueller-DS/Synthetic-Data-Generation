import shutil
import argparse


parser = argparse.ArgumentParser(description="name of the dataset")

parser.add_argument(
    "--dataset", type=str, required=True, help="Name of the dataset (e.g., 'adult')"
)

parser.add_argument(
    "--seed", type=str, required=True, help="Seed with the best performance"
)

args = parser.parse_args()
DATASET = args.dataset
SEED = args.seed

source_path = (
    f"data_generation/output_generator/{DATASET}/bgm/5_50/{SEED}/raw_gen_data.csv"
)
destination_path = f"../../data/synthetic/{DATASET}/vae-bgm.csv"

shutil.move(source_path, destination_path)
print(f"File moved from {source_path} to {destination_path}")
