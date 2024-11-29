import os
import logging
from datetime import datetime
import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    description="Evaulation Classifier"
)

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Name of the model to use (e.g., 'ctgan')"
)

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Name of the dataset (e.g., 'adult')"
)

parser.add_argument(
    "--run",
    type=str,
    required=True,
    help="Type of run"
)

args = parser.parse_args()


print(args.model)
print(args.dataset)
print(args.run)

# Set parmas
TS      = datetime.now().strftime("%Y%m%d_%H%M%S")
SDG     = args.model # name of used generation model 
RUN     = args.run # status of run test or final run 
DATASET = args.dataset # which dataset was used


# check if there is a log present
if not os.path.exists("results/resuls_df_log.csv"):
    RUN_ID = 1
else:
    id_old = pd.read_pickle("results/result_df_log.pkl")["ID"].to_list()[-1]
    RUN_ID = id_old + 1

print(RUN_ID)

### 
# logger definition
###
# set up logging
logger = logging.getLogger(__name__)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
formatter = logging.Formatter("%(levelname)s: %(message)s")
logger.setLevel(logging.INFO)

# log to stdout
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# log to file
log_file = f"logs/{RUN_ID}_{TS}_{RUN}_{SDG}_{DATASET}_log.txt"
print(log_file)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("-"*50)
logger.info(f"Timestamp: {TS}")
logger.info(f"Dataset: {DATASET}")
logger.info(f"Synthetic Data Generation Algo: {SDG}")
logger.info(f"Type of Run: {RUN}")


def sdg_run():
    print("sdg-run")

    # Path to data
    data_path_train = f"data/processed/{DATASET}/train.csv"
    syn_path_train = f"data/synthetic/{DATASET}/{SDG}.csv"
    data_path_test = f"data/processed/{DATASET}/test.csv" 

    # read data 
    df_train_ci = pd.read_csv(data_path_train)
    df_train_syn = pd.read_csv(syn_path_train)
    df_test = pd.read_csv(data_path_test)

    # combine syn and train
    df_train = pd.concat([df_train_ci, df_train_syn])

    print("Shape df_train:", df_train.shape)
    print("Shape df_test:", df_test.shape)



def baseline_run():
    pass


# check if baseline run or run with synthetic data
if SDG != "baseline":
    sdg_run()
else: baseline_run()