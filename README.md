# Synthetic-Data-Generation

Repo for bachelor Thesis

## Prerequisites

### Clone Repository

`git clone https://github.com/MarkusMueller-DS/Synthetic-Data-Generation.git`

### Create Python Environments

Here are the instruction on how to create the different environments.

#### General Environment

```
conda create -n sdg-env python=3.10
conda activate sdg-env
pip install -r requirements-sdg.txt
```

### TabSyn

- Instructions in Github repo: https://github.com/amazon-science/tabsyn
- the requirements.txt is in the folder `environments`

```
conda create -n tabsyn-env python=3.10
conda activate tabsyn-env
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements-tabsyn.txt
```

#### CTAB-GAN-Plus

- Github repo: https://github.com/Team-TUD/CTAB-GAN-Plus
- scikit-learn has to be installed via conda since version is no longer supported by pip
- the requirements.txt is in the folder `environments`

```
conda create -n ctab-gan-plus-env python=3.8
conda activate ctab-gan-plus-env
conda install -c conda-forge scikit-learn=0.24.1
pip install -r requirements-ctab-gan-plus.txt
```

#### VAE-BGM

- Github repo: https://github.com/Patricia-A-Apellaniz/vae-bgm_data_generator
- the requirements.txt is in the folder `environments`

```
conda create -n vae-bgm-env python=3.8
conda activate vae-bgm-env
pip install -r requirements-vae-bgm.txt
```

#### CTGAN & TVAE

- SDV Docu: https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers
- Python: 3.10.16

```
conda create -n sdv-env python=3.10
conda activate sdv-env
conda install -c pytorch -c conda-forge sdv
```

#### Evaluation

- Environment for evaluation code

```
conda create -n eval-env python=3.10
conda activate eval-env
conda install -c pytorch -c conda-forge sdv
pip install -r requirements-eval.txt
```

### Download Dataset

- raw data is downloaded in `data/raw/<name_of_datset>`

```
# 1. activate the general environment:
conda activate sdg-env

# 2. run download_datasets.py file specifing the dataset (adult, yeast, cc-fraud)
python download_dataset.py --dataset <name_of_dataset>
```

### Process Dataset

- processed data is saved in `data/processed/<name_of_dataset>`

```
# 1. activate the general environment:
conda activate sdg-env

# 2. run process_dataset.py file specifing the dataset(adult, yeast, cc-fraud)
python process_dataset.py --dataset <name_of_dataset>
```

## Create Synthetic Data

Steps to create synthetic data with the various models. The synthetic data is saved in `data/synthetic`. For baslien methods the resampled data is also safed in the same place.

Arguments:

- `<name_of_dataset>`: 'adult' 'yeast', 'cc-fraud'
- `<name_of_model>`: 'smote', 'ctgan', 'tabsyn', 'ctab-gan-plus', 'vae-bgm', 'tvae-all', 'tvae-top-2'

### Baseline

#### ROS & RUS

```
# 1. activate the general environemtn:
conda activate sdg-env

# 2. cd into the baseline folder
cd sdg-models/baseline

# 3. run over_under_sampling.py file specifing the dataset
python over_under_sampling.py --dataset <name_of_datset>
```

#### SMOTE

```
# 1. activate the general environemtn:
conda activate sdg-env

# 2. cd into the baseline folder
cd sdg-models/baseline

# 3. run smote.py file specifing the dataset
python smote.py --dataset <name_of_datset>
```

### SDG

#### CTGAN

```
# 1. activate the sdv environment:
conda activate sdv-env

# 2. cd into ctgan folder
cd sdg-models/ctgan

# 3. run main.py file specifing the dataset
python main.py --dataset <name_of_dataset>
```

#### CTAB-GAN-Plus

```
# 1. activate the ctab-gan-plus environment:
conda activate ctab-gan-plus-env

# 2. cd into ctab-gan-plus folder
cd sdg-models/ctab-gan-plus

# 3. run main.py file specifing the dataset
python main.py --dataset <name_of_dataset>
```

#### TabSyn

```
# 1. activate the tabsyn environment:
conda activate tabsyn-env

# 2. cd into tabsyn folder
cd sdg-models/tabsyn

# 3. train vae model
python main.py --dataname <name_of_dataset> --method vae --mode train

# 4. train diffusion model
python main.py --dataname <name_of_dataset> --method tabsyn --mode train

# 5. Sample data
python main.py --dataname <name_of_dataset> --method tabsyn --mode sample
```

#### VAE-BGM

- After synthetic data generation the best syhntetic data needs to be found since 15 different examples are created

```
# 1. activate the vae-bgm environment:
conda activate vae-bgm-env

# 2. cd into vae-bgm folder
cd sdg-models/vae-bgm

# 3. run main_genertor.py file in the folder data_generation specifing the dataset
python data_generation/main_generator --dataset <name_of_dataset>

# 4. find the best seed
python find_best_seed.py --dataset <name_of_dataset>

# 5. move synthetic data specifing the seed with the best performance from step 4
python move_syn_data.py --dataset <name_of_dataset> --seed <seed_XX>
```

#### TVAE

```
# 1. activate the sdv environment:
conda activate sdv-env

# 2. cd into tvae folder
cd sdg-models/tvae

# 3. run the python file for all and top-2 model specifing the dataset. When creating data for the cc-fraud dataset use the ci argument to indicate for which class imbalance. No need for the ci argument, for adult and yeast dataset
python main_all.py --dataset <name_of_dataset> --ci <1/5>
```

### Evaluation

#### Data Quality

- the results of the data quality evaluaiton are saved in `results/quality_data.csv`
- the evaluation script for the data quality needs to be run for every combination of dataset and model
- the `ci` argument is only relevant for the cc-fraud dataset
- for yeast `ctab-gan-plus` is not valid (no synthetic data)

```
# 1. activate the evaluation environment
conda activate eval-env

# 2. run the python file
python eval/eval_quality.py --dataset <name_of_dataset> --model <name_of_model> --ci <1/5>
```

#### Data Visualization

- visualizations are saved in `results/plots`
- the evaluation script for the data visualization needs to be run for every combination of dataset and model
- the `ci` argument is only relevant for the cc-fraud dataset
- column distribution plots and t-SNE visualizations are generated simultaneously but can be commented out if only one of them is required (line 408 & 409).
- for yeast `ctab-gan-plus` is not valid (no synthetic data)

```
# 1. activate the eval environment
conda activate eval-env

# 2. run the python file
python eval/eval_plots.py --dataset <name_of_dataset> --model <name_of_model> --ci <1/5>
```

#### Classification Performance

- the results of the data quality evaluaiton are saved in `results/`
- the evaluation script for the clf performance of the baseline needs to be run for every dataset
- the evaluation script for the clf performance of the different models needs to be run for every dataset and model
- the `ci` argument is only relevant for the cc-fraud dataset
- for yeast `ctab-gan-plus` is not valid (no synthetic data)

```
# 1. activate the eval environment
conda activate eval-env

# 2. run baseline classification
python eval/eval_clf_baseline.py --dataset <name_of_dataset> --ci <1/5>

# 3. run classification for the different models
python eval/eval_clf.py --dataset <name_of_dataset> --model <name_of_model> --ci <1/5>
```

## Folder Structure

- `data`
  - `info`: json files with information about the datsets
  - `processed`: processed datasets used for training and evaluation
  - `raw`: raw datasets
  - `synthetic`: generated data from the different methods and models
- `environments`: txt-files to build the various python environments
- `eval`: python scripts for evaluation
- `notebooks`: some EDA and evaluation notebooks used to test code
- `results`
  - `plots`: t-SNE and distributions of real and synthetic data
  - `tabele`: excel files containing the results of data quality and clf performance
  - csv and pickle file to save results
- `sdg-models`: Code for the different methods and models

## Misc files and scripts

- `eval/eval_create_xlsc.py`: creates excel tables for the thesis
- `zip_syn_data.py`: zips the synthetic data to better copy from GPU server
