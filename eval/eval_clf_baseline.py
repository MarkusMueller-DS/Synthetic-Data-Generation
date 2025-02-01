import sys
import os
import logging
from datetime import datetime
import pandas as pd
import argparse
import json

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier
import lightgbm as lgb

parser = argparse.ArgumentParser(description="Evaluation Classifier")

parser.add_argument(
    "--dataset", type=str, required=True, help="Name of the dataset (e.g., 'adult')"
)

parser.add_argument("--ci", type=str, default=None, help="Class imbalance ratio")

args = parser.parse_args()

CI = args.ci

print(args.dataset)

# Set parmas
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
SDG = "baseline"  # name of used generation model
DATASET = args.dataset  # which dataset was used


# check if there is a log present
if not os.path.exists("results/results_df_log.csv"):
    RUN_ID = 1
else:
    id_old = pd.read_pickle("results/results_df_log.pkl")["ID"].to_list()[-1]
    RUN_ID = id_old + 1

print(RUN_ID)

# load dataset info JSON
info_path = f"data/info/{DATASET}.json"
if not os.path.exists(info_path):
    raise FileNotFoundError(f"The file does not exists")
else:
    print("info.json found")

# load info json
with open(info_path, "r") as f:
    info = json.load(f)


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
if CI != None:
    log_file = f"logs/{RUN_ID}_{TS}_{SDG}_{DATASET}-{CI}_log.txt"
else:
    log_file = f"logs/{RUN_ID}_{TS}_{SDG}_{DATASET}_log.txt"

print(log_file)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("-" * 50)
logger.info(f"Timestamp: {TS}")
if CI != None:
    logger.info(f"Dataset: {DATASET}-{CI}")
else:
    logger.info(f"Dataset: {DATASET}")
logger.info(f"Synthetic Data Generation Algo: {SDG}")


def impute_missing_values(df):
    # Category: repalce nan with "missing"
    cat_columns_idx = info["cat_col_idx"]
    columns_names = info["column_names"]
    cat_columns_names = [columns_names[i] for i in cat_columns_idx]

    for col in cat_columns_names:
        df[col] = df[col].fillna("Missing")

    return df


def apply_onehot_encoding(train_df, test_df, categorical_columns):
    # Create OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # Apply ColumnTransformer to the categorical columns
    preprocessor = ColumnTransformer(
        transformers=[(col, encoder, [col]) for col in categorical_columns],
        remainder="passthrough",  # Keep non-categorical columns unchanged
    )

    # Fit the encoder on the train data and transform both train and test data
    train_encoded = preprocessor.fit_transform(train_df)
    test_encoded = preprocessor.transform(test_df)

    # Convert the encoded data back to DataFrame
    train_encoded_df = pd.DataFrame(
        train_encoded, columns=preprocessor.get_feature_names_out()
    )
    test_encoded_df = pd.DataFrame(
        test_encoded, columns=preprocessor.get_feature_names_out()
    )

    # Return the encoded DataFrames
    return train_encoded_df, test_encoded_df


def baseline_run():
    if CI != None:
        print(f"Baseline run for dataset: {DATASET}-{CI}")
    else:
        print(f"Baseline run for dataset: {DATASET}")

    # Path to data
    if CI != None:
        data_path_train = f"data/processed/{DATASET}/train_src_{CI}.csv"
    else:
        data_path_train = f"data/processed/{DATASET}/train_src.csv"
    data_path_test = f"data/processed/{DATASET}/test.csv"

    # read data
    df_train = pd.read_csv(data_path_train)
    df_test = pd.read_csv(data_path_test)

    target_col_name = info["target_col"]
    print("Target column name:", target_col_name)

    counts = df_train[target_col_name].value_counts()

    # log dataset infos
    logger.info("-" * 50)
    logger.info("Dataset infos")
    logger.info(counts)
    logger.info(f"Shape df_train: {df_train.shape}")
    logger.info(f"Shape df_test: {df_test.shape}")

    # pre process
    # certain classification models need numbers as the y variable, like XGBoost
    # 1 for minority class and 0 for majority class
    df_train[target_col_name] = df_train[target_col_name].map(
        {info["majority_class"]: 0, info["minority_class"]: 1}
    )
    df_test[target_col_name] = df_test[target_col_name].map(
        {info["majority_class"]: 0, info["minority_class"]: 1}
    )

    # impute misisng values
    # yeast has no missing values
    if DATASET in ["adult"]:
        df_train = impute_missing_values(df_train)
        df_test = impute_missing_values(df_test)

    print("NaNs in df_train:", df_train.isnull().values.any())
    print("NaNs in df_test:", df_test.isnull().values.any())

    # get names of the categorical column for one hot encoding
    if DATASET in ["adult"]:
        cat_columns_idx = info["cat_col_idx"]
        columns_names = info["column_names"]
        cat_columns_names = [columns_names[i] for i in cat_columns_idx]
        # on hot encode categorical data
        df_train, df_test = apply_onehot_encoding(df_train, df_test, cat_columns_names)

        # split data with onehot encoding
        X_train = df_train.drop(columns=[f"remainder__{target_col_name}"])
        y_train = df_train[f"remainder__{target_col_name}"]

        X_test = df_test.drop(columns=[f"remainder__{target_col_name}"])
        y_test = df_test[f"remainder__{target_col_name}"]
    else:
        # split data without one hot encoding
        X_train = df_train.drop(columns=[target_col_name])
        y_train = df_train[target_col_name]

        X_test = df_test.drop(columns=[target_col_name])
        y_test = df_test[target_col_name]

    logger.info("-" * 50)
    logger.info("Shape of train and test data")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    logger.info(f"X_train: {X_train.shape}")
    logger.info(f"y_train: {y_train.shape}")
    logger.info(f"X_test: {X_test.shape}")
    logger.info(f"y_test: {y_test.shape}")

    logger.info("-" * 50)
    logger.info(
        f"Stating training of classifier: {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    clfs = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42, algorithm="SAMME"),
        "LightGBM": lgb.LGBMClassifier(random_state=42),
        "KNeighbors": KNeighborsClassifier(),
        # "SVM": SVC(random_state=42),
    }

    logger.info("Classification Models:")
    for name, clf in clfs.items():
        logger.info(f"{name}: {clf.get_params()}")

    # save f1_socres in results dict
    results = {}

    for name, clf in clfs.items():
        logger.info(f"Results for {name}")
        # train model
        clf.fit(X_train, y_train)

        # predcit
        y_pred = clf.predict(X_test)

        # calcualte f1_score
        accuracy_score_value = accuracy_score(y_test, y_pred)
        f1_score_value = float(f1_score(list(y_test), list(y_pred), pos_label=1))
        roc_auc_score_value = float(roc_auc_score(y_test, y_pred))

        # log results
        logger.info(f"Accuracy Score: {accuracy_score_value}")
        logger.info(f"F1-Score of minority class: {f1_score_value}")
        logger.info(f"Roc-AUC-Score: {roc_auc_score_value}")
        logger.info(f"{classification_report(y_test, y_pred)}")

        # add results to dict
        results[name] = (accuracy_score_value, f1_score_value, roc_auc_score_value)

    logger.info(
        f"Finished training of classifier: {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    logger.info("-" * 50)
    logger.info("Saving results...")

    # add ci label when dataset is cc-fraud
    dataset = DATASET
    if dataset == "cc-fraud":
        dataset += f"-{CI}"

    if not os.path.exists("results/results_df_log.csv"):
        # init pandas df and save
        df_save = pd.DataFrame(
            [
                {
                    "ID": 0,
                    "Timestamp": TS,
                    "SDG": SDG,
                    "Dataset": dataset,
                    "LR_accuracy": round(results["Logistic Regression"][0] * 100, 2),
                    "LR_f1-score": round(results["Logistic Regression"][1] * 100, 2),
                    "LR_roc_auc": round(results["Logistic Regression"][2] * 100, 2),
                    "RF_accuracy": round(results["Random Forest"][0] * 100, 2),
                    "RF_f1-score": round(results["Random Forest"][1] * 100, 2),
                    "RF_roc_auc": round(results["Random Forest"][2] * 100, 2),
                    "XGB_accuracy": round(results["XGBoost"][0] * 100, 2),
                    "XGB_f1-score": round(results["XGBoost"][1] * 100, 2),
                    "XGB_roc_auc": round(results["XGBoost"][2] * 100, 2),
                    "Ada_accuracy": round(results["AdaBoost"][0] * 100, 2),
                    "Ada_f1-score": round(results["AdaBoost"][1] * 100, 2),
                    "Ada_roc_auc": round(results["AdaBoost"][2] * 100, 2),
                    "LGBM_accuracy": round(results["LightGBM"][0] * 100, 2),
                    "LGBM_f1-score": round(results["LightGBM"][1] * 100, 2),
                    "LGBM_roc_auc": round(results["LightGBM"][2] * 100, 2),
                    "KNN_accuracy": round(results["KNeighbors"][0] * 100, 2),
                    "KNN_f1-score": round(results["KNeighbors"][1] * 100, 2),
                    "KNN_roc_auc": round(results["KNeighbors"][2] * 100, 2),
                    # "SVM_accuracy": results["SVM"][0],
                    # "SVM_f1-score": results["SVM"][1],
                    # "SVM_roc_auc": results["SVM"][2],
                }
            ]
        )

        df_save.to_pickle("results/results_df_log.pkl")
        df_save.to_csv("results/results_df_log.csv", index=False)
    else:
        # read pandas and append results
        df = pd.read_csv("results/results_df_log.csv")

        df_new = pd.DataFrame(
            [
                {
                    "ID": RUN_ID,
                    "Timestamp": TS,
                    "SDG": SDG,
                    "Dataset": dataset,
                    "LR_accuracy": round(results["Logistic Regression"][0] * 100, 2),
                    "LR_f1-score": round(results["Logistic Regression"][1] * 100, 2),
                    "LR_roc_auc": round(results["Logistic Regression"][2] * 100, 2),
                    "RF_accuracy": round(results["Random Forest"][0] * 100, 2),
                    "RF_f1-score": round(results["Random Forest"][1] * 100, 2),
                    "RF_roc_auc": round(results["Random Forest"][2] * 100, 2),
                    "XGB_accuracy": round(results["XGBoost"][0] * 100, 2),
                    "XGB_f1-score": round(results["XGBoost"][1] * 100, 2),
                    "XGB_roc_auc": round(results["XGBoost"][2] * 100, 2),
                    "Ada_accuracy": round(results["AdaBoost"][0] * 100, 2),
                    "Ada_f1-score": round(results["AdaBoost"][1] * 100, 2),
                    "Ada_roc_auc": round(results["AdaBoost"][2] * 100, 2),
                    "LGBM_accuracy": round(results["LightGBM"][0] * 100, 2),
                    "LGBM_f1-score": round(results["LightGBM"][1] * 100, 2),
                    "LGBM_roc_auc": round(results["LightGBM"][2] * 100, 2),
                    "KNN_accuracy": round(results["KNeighbors"][0] * 100, 2),
                    "KNN_f1-score": round(results["KNeighbors"][1] * 100, 2),
                    "KNN_roc_auc": round(results["KNeighbors"][2] * 100, 2),
                    # "SVM_accuracy": results["SVM"][0],
                    # "SVM_f1-score": results["SVM"][1],
                    # "SVM_roc_auc": results["SVM"][2],
                }
            ]
        )

        df_save = pd.concat([df, df_new])
        df_save.to_pickle("results/results_df_log.pkl")
        df_save.to_csv("results/results_df_log.csv", index=False)

    logger.info("Results saved")


baseline_run()
