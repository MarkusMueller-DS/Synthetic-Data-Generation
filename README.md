# Synthetic-Data-Generation

Repo for bachelor Thesis

## How to run

### 1. Downlaod data

`python download_datasets.py --dataset adult`

### 2. Process data

`python process_datasets.py --dataset adult`

### 3. Train SDG-Models

`python main.py --dataset adult --model ctgan --mode train`

### 4. Sample SDG-Models

`python main.py --dataset adult --model ctgan --mode sample`

### 5. Evalution with classification models

`python eval/eval.py --dataset adult --model ctgan --run test`
