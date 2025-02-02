# Synthetic-Data-Generation

Repo for bachelor Thesis

## Prerequisites

### Clone Repository

`git clone https://github.com/MarkusMueller-DS/Synthetic-Data-Generation.git`

### Create Python Environments

Here are the instruction on how to create the different environments. I tested to expport the different environemtns as a yml file and and then create a new environment from that file but there were multipe errors regards the pip packages so here is a step by step process for every environment needed.

#### General Environment

### TabSyn

- Instructions in Github repo: https://github.com/amazon-science/tabsyn
- the relevant requirements.txt is in the folder `sdg-models/tabsyn/`

```
conda create -n tabsyn-env python=3.10
conda activate tabsyn-env
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

#### CTAB-GAN-Plus

- Github repo: https://github.com/Team-TUD/CTAB-GAN-Plus
- the requirements.txt is in the folder `sdg-models/ctab-gan-plus`

```
conda create -n ctab-gan-plus-env python=3.8
conda activate ctab-gan-plus-env

```

#### VAE-BGM

- Github repo: https://github.com/Patricia-A-Apellaniz/vae-bgm_data_generator
- Python: 3.8.20
- Environment: `environment-vae-bgm.yml`

#### CTGAN & TVAE

- SDV Docu: https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers
- Python: 3.10.16
- Environment: `environment-sdv.yml`

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

1. activate the sdv environment:
   `conda activate ctgan-env`

2. cd into ctgan folder
   `cd sdg-models/ctgan`

3. run main.py file specifing the dataset
   `python main.py --dataset <name_of_dataset>`

#### CTAB-GAN-Plus

1. activate the ctab-gan-plus environment:
   `conda activate ctab-gan-plus-env`

2. cd into ctab-gan-plus folder
   `cd sdg-models/ctab-gan-plus`

3. run main.py file specifing the dataset
   `python main.py --dataset <name_of_dataset>`

#### TabSyn

1. activate the tabsyn environment:
   `conda activate tabsyn-env`

2. cd into ctab-gan-plus folder
   `cd sdg-models/tabsyn`

3. run main.py file specifing the dataset
   `python main.py --dataset <name_of_dataset>`

#### VAE-BGM

- After synthetic data generation the best syhntetic data needs to be found since 15 different examples are created

1. activate the vae-bgm environment:
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

1. activate the sdv environment:
   `conda activate ctgan-env`

2. cd into ctab-gan-plus folder
   `cd sdg-models/TVAE`

3. run the python file for all and top-2 model specifing the dataset. When creating data for the cc-fraud dataset use the ci argument to indicate for which class imbalance. No need for the ci argument, for adult and yeast dataset
   `python main_all.py --dataset <name_of_dataset> --ci <1/5>`

### Evaluation

#### Data Quality

- the results of the data quality evaluaiton are saved in `results/quality_data.csv`
- the evaluation script for the data quality needs to be run for every combination of dataset and model
- the `ci` argument is only relevant for the cc-fraud dataset
- `model` argument can be: `smote`, `ctgan`, `ctab-gan-plus`, `tabsyn`, `vae-bgm`, `tvae-all` and `tave-top-2`
- for yeast `ctab-gan-plus` is not valid (no synthetic data)

1. activate the general environment
   `conda activate sdg-eda-env`

2. run the python file
   `python eval/eval_quality.py --dataset <name_of_dataset> --model <name_of_model> --ci <1/5>`

#### Data Visualization

- visualizations are saved in `results/plots`
- the evaluation script for the data visualization needs to be run for every combination of dataset and model
- the `ci` argument is only relevant for the cc-fraud dataset
- column distribution plots and t-SNE visualizations are generated simultaneously but can be commented out if only one of them is required (line 408 & 409).
- `model` argument can be: `smote`, `ctgan`, `ctab-gan-plus`, `tabsyn`, `vae-bgm`, `tvae-all` and `tave-top-2`
- for yeast `ctab-gan-plus` is not valid (no synthetic data)

1. activate the general environment
   `conda activate sdg-eda-env`

2. run the python file
   `python eval/eval_plots.py --dataset <name_of_dataset> --model <name_of_model> --ci <1/5>`

#### Classification Performance

- the results of the data quality evaluaiton are saved in `results/`
- the evaluation script for the clf performance of the baseline needs to be run for every dataset
- the evaluation script for the clf performance of the different models needs to be run for every dataset and model
- the `ci` argument is only relevant for the cc-fraud dataset
- `model` argument can be: `smote`, `ctgan`, `ctab-gan-plus`, `tabsyn`, `vae-bgm`, `tvae-all` and `tave-top-2`
- for yeast `ctab-gan-plus` is not valid (no synthetic data)

1. activate the general environment
   `conda activate sdg-eda-env`

2. run baseline classification
   `python eval/eval_clf_baseline.py --dataset <name_of_dataset> --ci <1/5>`

3. run classification for the different models
   `python eval/eval_clf.py --dataset <name_of_dataset> --model <name_of_model> --ci <1/5>`

## Misc files and scripts

- `eval/eval_create_xlsc.py`: creates excel tables for the thesis
- `zip_syn_data.py`: zips the synthetic data to better copy from GPU server
- `data/info`: relevant information of the different datasets (important)
