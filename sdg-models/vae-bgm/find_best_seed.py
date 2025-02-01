import sys
import os
import pandas as pd
import argparse
import json

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description="Find best seed")

parser.add_argument(
    "--dataset", type=str, required=True, help="Name of the dataset (e.g., 'adult')"
)

args = parser.parse_args()

DATASET = args.dataset

info_path = f"../../data/info/{DATASET}.json"
with open(info_path, "r") as f:
    info = json.load(f)

target_col = info["target_col"]
minority_class = info["minority_class"]


def impute_missing_values(df, info):
    # Category: repalce nan with "missing"
    cat_columns = info["cat_col_names"]
    num_columns = info["num_col_names"]

    for col in cat_columns:
        df[col] = df[col].fillna("Missing")
    for col in num_columns:
        df[col] = df[col].fillna(0.0)

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


def clf(syn_data, train_src, df_test_original):
    df_test = df_test_original.copy()

    # add synthetic data of minority class to train_ci dataset
    data_minority = syn_data[syn_data[target_col] == minority_class]
    # concat data
    df_train = pd.concat([train_src, data_minority])

    df_train[target_col] = df_train[target_col].map(
        {info["majority_class"]: 0, info["minority_class"]: 1}
    )
    df_test[target_col] = df_test[target_col].map(
        {info["majority_class"]: 0, info["minority_class"]: 1}
    )

    cat_columns_idx = info["cat_col_idx"]
    columns_names = info["column_names"]
    cat_columns_names = [columns_names[i] for i in cat_columns_idx]

    if DATASET in ["adult"]:
        # impute misisng values
        df_train = impute_missing_values(df_train, info)
        df_test = impute_missing_values(df_test, info)
        # on hot encode categorical data
        df_train, df_test = apply_onehot_encoding(df_train, df_test, cat_columns_names)

        # print("NaNs in df_train:", df_train.isnull().values.any())
        # print("NaNs in df_test:", df_test.isnull().values.any())

        # split data
        X_train = df_train.drop(columns=[f"remainder__{target_col}"])
        y_train = df_train[f"remainder__{target_col}"]

        X_test = df_test.drop(columns=[f"remainder__{target_col}"])
        y_test = df_test[f"remainder__{target_col}"]
    else:
        # print("NaNs in df_train:", df_train.isnull().values.any())
        # print("NaNs in df_test:", df_test.isnull().values.any())
        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]

        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]

    rf_clf = RandomForestClassifier(random_state=42)

    # fit model
    rf_clf.fit(X_train, y_train)

    # predict
    y_pred = rf_clf.predict(X_test)

    f1_score_value = float(f1_score(list(y_test), list(y_pred), pos_label=1))

    return f1_score_value


def find_best_seed():

    # path to syn data folder:
    syn_fodler_path = f"data_generation/output_generator/{DATASET}/bgm/5_50"

    seed_list = [
        "seed_0",
        "seed_1",
        "seed_2",
        "seed_3",
        "seed_4",
        "seed_5",
        "seed_6",
        "seed_7",
        "seed_8",
        "seed_9",
        "seed_10",
        "seed_11",
        "seed_12",
        "seed_13",
        "seed_14",
    ]

    data_path_src = f"../../data/processed/{DATASET}/train_src.csv"
    data_path_test = f"../../data/processed/{DATASET}/test.csv"

    df_train_ci = pd.read_csv(data_path_src)
    df_test = pd.read_csv(data_path_test)

    result_dict = {}

    for seed in seed_list:
        # read syn data for seed
        df_syn = pd.read_csv(f"{syn_fodler_path}/{seed}/raw_gen_data.csv")

        # do clf
        result = clf(df_syn, df_train_ci, df_test)
        print(f"{seed} f1_score: {result}")
        result_dict[seed] = result

    best_seed = max(result_dict, key=result_dict.get)
    print("best seed:", best_seed)


find_best_seed()
