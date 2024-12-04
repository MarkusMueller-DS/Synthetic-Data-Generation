# Synthetic-Data-Generation

Repo for bachelor Thesis

## How to run

### 1. Downlaod data

`python download_datasets.py`

### 2. Create dataset info json

Create a basic json file with the following information and place it into `data/info/<NAME_OF_DATASET>.json`

```
{
    "name": "<NAME_OF_DATASET>",
    "header": "infer",
    "column_names": [LIST],
    "num_col_idx": [LIST],  # list of indices of numerical columns
    "cat_col_idx": [LIST],  # list of indices of categorical columns
    "target_col_idx": <int>, # indice of the target column (for MLE)
    "file_type": "csv",
    "data_path": "data/<NAME_OF_DATASET>/<NAME_OF_DATASET>.csv"
    "test_path": null,
}
```

### 3. Process data

`python process_datasets.py --dataset adult`

During process data the info json is used to process the dataset and a new json is created that is used for the process, with additional information

### 4. Train SDG-Models

`python main.py --dataset adult --model ctgan --mode train`

### 5. Sample SDG-Models

`python main.py --dataset adult --model ctgan --mode sample`

### 6. Evalution with classification models

`python eval/eval.py --dataset adult --model ctgan --run test`
