import pandas as pd


def clf_performance():
    clf_result_path = "results/results_df_log.csv"
    df_clf = pd.read_csv(clf_result_path)
    df_clf.drop(columns=["ID", "Timestamp"], inplace=True)

    # aggregate scores for clf performance
    accuracy_columns = [col for col in df_clf.columns if "accuracy" in col]
    f1_score_columns = [col for col in df_clf.columns if "f1-score" in col]
    roc_auc_columns = [col for col in df_clf.columns if "roc_auc" in col]
    # transform to float
    df_clf[accuracy_columns] = (
        df_clf[accuracy_columns].replace({",": "."}, regex=True).astype(float)
    )
    df_clf[f1_score_columns] = (
        df_clf[f1_score_columns].replace({",": "."}, regex=True).astype(float)
    )
    df_clf[roc_auc_columns] = (
        df_clf[roc_auc_columns].replace({",": "."}, regex=True).astype(float)
    )

    accuracy_avg = df_clf[accuracy_columns].mean(axis=1).round(2)
    f1_score_avg = df_clf[f1_score_columns].mean(axis=1).round(2)
    roc_auc_avg = df_clf[roc_auc_columns].mean(axis=1).round(2)

    df_clf["accuracy_avg"] = accuracy_avg
    df_clf["f1_score_avg"] = f1_score_avg
    df_clf["roc_auc_avg"] = roc_auc_avg

    # create excel files
    df_clf[df_clf["Dataset"] == "adult"].to_excel(
        "results/tabels/adult.xlsx", index=False
    )
    df_clf[df_clf["Dataset"] == "yeast"].to_excel(
        "results/tabels/yeast.xlsx", index=False
    )
    df_clf[df_clf["Dataset"] == "cc-fraud-1"].to_excel(
        "results/tabels/cc-fraud-1.xlsx", index=False
    )
    df_clf[df_clf["Dataset"] == "cc-fraud-5"].to_excel(
        "results/tabels/cc-fraud-5.xlsx", index=False
    )

    print("Created excel files for clf performance")


def data_quality():
    quality_result_path = "results/quality_data.csv"
    df_quality = pd.read_csv(quality_result_path)

    for col in df_quality.select_dtypes(include=["float"]).columns:
        df_quality[col] = df_quality[col].map(lambda x: f"{x:.4f}".replace(".", ","))
    # creat excel files
    df_quality[df_quality["Dataset"] == "adult"].to_excel(
        "results/tabels/adult_quality.xlsx", index=False
    )
    df_quality[df_quality["Dataset"] == "yeast"].to_excel(
        "results/tabels/yeast_quality.xlsx", index=False
    )
    df_quality[df_quality["Dataset"] == "cc-fraud-1"].to_excel(
        "results/tabels/cc-fraud-1_quality.xlsx", index=False
    )
    df_quality[df_quality["Dataset"] == "cc-fraud-5"].to_excel(
        "results/tabels/cc-fraud-5_quality.xlsx", index=False
    )

    print("Created excel files for data quality")


data_quality()
clf_performance()
