# Synthetic-Data-Generation

Repo for bachelor Thesis

## Prerequisites

### Clone Repository

`git clone https://github.com/MarkusMueller-DS/Synthetic-Data-Generation.git`

### Create Python Environments

The environemt ymls are in the folder `environments`.
Create Conda environement (replace the yml file for each environemtn):
`conda env create -f environemt.yml`

#### General Environment

Python: 3.10.15
Environment: `environment-sdg.yml`

### TabSyn

Github repo: https://github.com/amazon-science/tabsyn
Python: 3.10.15
Environment: `environment-tabsyn.yml`

#### CTAB-GAN-Plus

Github repo: https://github.com/Team-TUD/CTAB-GAN-Plus
Python: 3.8.20
Environment: `environment-ctab.yml`

#### VAE-BGM

Github repo: https://github.com/Patricia-A-Apellaniz/vae-bgm_data_generator
Python: 3.8.20
Environment: `environment-vae-bgm.yml`

#### CTGAN & TVAE

Python: 3.10.16
Environment: `environment-sdv.yml`

### Download Dataset

- raw data is downloaded in `data/raw/<name_of_datset>`

1. activate the general environment:
   `conda activate sdg-eda-env`

2. run download_datasets.py file specifing the dataset (adult, yeast, cc-fraud)
   `python download_dataset.py --dataset <name_of_dataset>`

### Process Dataset

- processed data is saved in `data/processed/<name_of_dataset>`

1. activate the general environment:
   `conda activate sdg-eda-env`

2. run process_dataset.py file specifing the dataset(adult, yeast, cc-fraud)
   `python process_dataset.py --dataset <name_of_dataset>`

## Create Synthetic Data

Steps to create synthetic data with the various models. The synthetic data is saved in `data/synthetic`. For baslien methods the resampled data is also safed in the same place.

### Baseline

#### ROS & RUS

1. activate the general environemtn:
   `conda activate sdg-eda-env`

2. cd into the baseline folder
   `cd sdg-models/baseline`

3. run over_under_sampling.py file specifing the dataset
   `python over_under_sampling.py --dataset <name_of_datset>`

#### SMOTE

1. activate the general environemtn:
   `conda activate sdg-eda-env`

2. cd into the baseline folder
   `cd sdg-models/baseline`

3. run smote.py file specifing the dataset
   `python smote.py --dataset <name_of_datset>`

### SDG

#### CTGAN

1. activate the sdv environemtn:
   `conda activate ctgan-env`

2. cd into ctgan folder
   `cd sdg-models/ctgan`

3. run main.py file specifing the dataset
   `python main.py --dataset <name_of_dataset>`

#### CTAB-GAN-Plus

1. activate the ctab-gan-plus environemtn:
   `conda activate ctab-gan-plus-env`

2. cd into ctab-gan-plus folder
   `cd sdg-models/ctab-gan-plus`

3. run main.py file specifing the dataset
   `python main.py --dataset <name_of_dataset>`

#### TabSyn

1. activate the tabsyn environemtn:
   `conda activate tabsyn-env`

2. cd into ctab-gan-plus folder
   `cd sdg-models/tabsyn`

3. run main.py file specifing the dataset
   `python main.py --dataset <name_of_dataset>`

#### VAE-BGM

- After synthetic data generation the best syhntetic data needs to be found since 15 different examples are created

1. activate the vae-bgm environemtn:
   `conda activate vae-bgm-env`

2. cd into ctab-gan-plus folder
   `cd sdg-models/vae-bgm`

3. run main_genertor.py file in the folder data_generation specifing the dataset
   `python data_generation/main_generator --dataset <name_of_dataset>`

4. find the best seed
   `python find_best_seed.py --dataset <name_of_dataset>`

5. move synthetic data specifing the seed with the best performance from step 4
   `python move_syn_data.py --dataset <name_of_dataset> --seed <best_seed>`

#### TVAE

1. activate the sdv environemtn:
   `conda activate ctgan-env`

2. cd into ctab-gan-plus folder
   `cd sdg-models/TVAE`

3. run the python file for all and top-2 model specifing the dataset. When creating data for the cc-fraud dataset use the ci argument to indicate for which class imbalance. No need for the ci argument, for adult and yeast dataset
   `python main_all.py --dataset <name_of_dataset> --ci <1/5>`

### Evaluation

#### Data Quality

#### Data Visualization

#### Classification Performance
