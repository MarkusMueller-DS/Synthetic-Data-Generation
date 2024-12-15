import json
import pandas as pd
import numpy as np
from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader
from sklearn.preprocessing import OneHotEncoder
import os

real_path = "../sdg-models/tabsyn/synthetic/adult/real.csv"
syn_path = "../sdg-models/tabsyn/synthetic/adult/tabsyn.csv"

real_data = pd.read_csv(real_path)
syn_data = pd.read_csv(syn_path)

# load info json
with open("../sdg-models/tabsyn/data/info/adult.json", "r") as f:
    info = json.load(f)

dataname = "adult"
model = "tabsyn"

real_data.columns = range(len(real_data.columns))
syn_data.columns = range(len(syn_data.columns))

num_col_idx = info["num_col_idx"]
cat_col_idx = info["cat_col_idx"]
target_col_idx = info["target_col_idx"]
cat_col_idx += target_col_idx

num_real_data = real_data[num_col_idx]
cat_real_data = real_data[cat_col_idx]

num_real_data_np = num_real_data.to_numpy()
cat_real_data_np = cat_real_data.to_numpy().astype("str")


num_syn_data = syn_data[num_col_idx]
cat_syn_data = syn_data[cat_col_idx]

num_syn_data_np = num_syn_data.to_numpy()

# cat_syn_data_np = np.array
cat_syn_data_np = cat_syn_data.to_numpy().astype("str")
if (dataname == "default" or dataname == "news") and model[:4] == "codi":
    cat_syn_data_np = cat_syn_data.astype("int").to_numpy().astype("str")

elif model[:5] == "great":
    if dataname == "shoppers":
        cat_syn_data_np[:, 1] = cat_syn_data[11].astype("int").to_numpy().astype("str")
        cat_syn_data_np[:, 2] = cat_syn_data[12].astype("int").to_numpy().astype("str")
        cat_syn_data_np[:, 3] = cat_syn_data[13].astype("int").to_numpy().astype("str")

        max_data = cat_real_data[14].max()

        cat_syn_data.loc[cat_syn_data[14] > max_data, 14] = max_data
        # cat_syn_data[14] = cat_syn_data[14].apply(lambda x: threshold if x > max_data else x)

        cat_syn_data_np[:, 4] = cat_syn_data[14].astype("int").to_numpy().astype("str")
        cat_syn_data_np[:, 4] = cat_syn_data[14].astype("int").to_numpy().astype("str")

    elif dataname in ["default", "faults", "beijing"]:

        columns = cat_real_data.columns
        for i, col in enumerate(columns):
            if cat_real_data[col].dtype == "int":

                max_data = cat_real_data[col].max()
                min_data = cat_real_data[col].min()

                cat_syn_data.loc[cat_syn_data[col] > max_data, col] = max_data
                cat_syn_data.loc[cat_syn_data[col] < min_data, col] = min_data

                cat_syn_data_np[:, i] = (
                    cat_syn_data[col].astype("int").to_numpy().astype("str")
                )

    else:
        cat_syn_data_np = cat_syn_data.to_numpy().astype("str")

else:
    cat_syn_data_np = cat_syn_data.to_numpy().astype("str")

encoder = OneHotEncoder()
encoder.fit(cat_real_data_np)


cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()

le_real_data = pd.DataFrame(
    np.concatenate((num_real_data_np, cat_real_data_oh), axis=1)
).astype(float)
le_real_num = pd.DataFrame(num_real_data_np).astype(float)
le_real_cat = pd.DataFrame(cat_real_data_oh).astype(float)


le_syn_data = pd.DataFrame(
    np.concatenate((num_syn_data_np, cat_syn_data_oh), axis=1)
).astype(float)
le_syn_num = pd.DataFrame(num_syn_data_np).astype(float)
le_syn_cat = pd.DataFrame(cat_syn_data_oh).astype(float)

np.set_printoptions(precision=4)

result = []

print("=========== All Features ===========")
print("Data shape: ", le_syn_data.shape)

print(le_syn_data)
print(le_real_data)

X_syn_loader = GenericDataLoader(le_syn_data)
X_real_loader = GenericDataLoader(le_real_data)

quality_evaluator = eval_statistical.AlphaPrecision()
qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
qual_res = {
    k: v for (k, v) in qual_res.items() if "naive" in k
}  # use the naive implementation of AlphaPrecision
qual_score = np.mean(list(qual_res.values()))

print(
    "alpha precision: {:.6f}, beta recall: {:.6f}".format(
        qual_res["delta_precision_alpha_naive"], qual_res["delta_coverage_beta_naive"]
    )
)

Alpha_Precision_all = qual_res["delta_precision_alpha_naive"]
Beta_Recall_all = qual_res["delta_coverage_beta_naive"]


print(Alpha_Precision_all)
print(Beta_Recall_all)
