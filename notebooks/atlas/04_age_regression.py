from pytorch_tabular.utils import load_covertype_dataset
from rich.pretty import pprint
from sklearn.model_selection import BaseCrossValidator, ParameterGrid, ParameterSampler
import torch
import pickle
import shutil
import shap
from sklearn.model_selection import RepeatedStratifiedKFold
from glob import glob
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.model_selection import train_test_split
import numpy as np
from pytorch_tabular.utils import make_mixed_dataset, print_metrics
from pytorch_tabular import available_models
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, GANDALFConfig, TabNetModelConfig, FTTransformerConfig, DANetConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.tabular_model_tuner import TabularModelTuner
from torchmetrics.functional.regression import mean_absolute_error, pearson_corrcoef
from pytorch_tabular import MODEL_SWEEP_PRESETS
import pandas as pd
from pytorch_tabular import model_sweep
from src.pt.model_sweep import model_sweep_custom
import warnings
from src.utils.configs import read_parse_config
from src.utils.hash import dict_hash
from src.pt.hyper_opt import train_hyper_opt
import pathlib
from tqdm import tqdm
import distinctipy
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from statannotations.Annotator import Annotator
from scipy.stats import mannwhitneyu
import optuna
import shutup


shutup.please()


# Load data

feats_set = 'inbody'

path = f"D:/YandexDisk/Work/bbd/atlas/subset_{feats_set}"
path_configs = "D:/Work/bbs/notebooks/atlas/configs"
path_ckpts = "D:/Work/bbs/notebooks/atlas/pt"

tst_n_splits = 5
tst_n_repeats = 5
tst_random_state = 1337
tst_split_id = 1

val_n_splits = 4
val_n_repeats = 4
val_random_state = 1337
val_fold_id = 5

data = pd.read_excel(f"{path}/data.xlsx", index_col=0)
df_feats = pd.read_excel(f"{path}/feats.xlsx", index_col=0)
feats_ru_2_en = dict(zip(df_feats.index, df_feats['English']))
feats_en_2_ru = dict(zip(df_feats['English'], df_feats.index))
feat_trgt = 'Возраст'
feats_cnt = np.sort(df_feats.index[df_feats['Type'] == 'continuous'].to_list())
feats_cnt = list(feats_cnt[feats_cnt != 'Возраст'])
feats_cat = df_feats.index[df_feats['Type'] == 'categorical'].to_list()
feats = list(feats_cnt) + list(feats_cat)

feats_cnt = [feats_ru_2_en[x] for x in feats_cnt]
feats_cat = [feats_ru_2_en[x] for x in feats_cat]
feats = [feats_ru_2_en[x] for x in feats]
feat_trgt = feats_ru_2_en[feat_trgt]
data.rename(columns=feats_ru_2_en, inplace=True)

# Load stratification

with open(f"{path}/samples_tst({tst_random_state}_{tst_n_splits}_{tst_n_repeats})_val({val_random_state}_{val_n_splits}_{val_n_repeats}).pickle", 'rb') as handle:
    samples = pickle.load(handle)

for split_id in range(tst_n_splits * tst_n_repeats):
    for fold_id in range(val_n_splits * val_n_repeats):
        test_samples = samples[split_id]['test']
        train_samples = samples[split_id]['trains'][fold_id]
        validation_samples = samples[split_id]['validations'][fold_id]

        intxns = {
            'train_validation': set.intersection(set(train_samples), set(validation_samples)),
            'validation_test': set.intersection(set(validation_samples), set(test_samples)),
            'train_test': set.intersection(set(train_samples), set(test_samples))
        }
        
        for intxn_name, intxn_samples in intxns.items():
            if len(intxn_samples) > 0:
                print(f"Non-zero {intxn_name} intersection ({len(intxn_samples)}) for {split_id} Split and {fold_id} Fold!")
                
# Train, Validation, Test selection

split_dict = samples[tst_split_id]

test = data.loc[split_dict['test'], feats + [feat_trgt]]
train = data.loc[split_dict['trains'][val_fold_id], feats + [feat_trgt]]
validation = data.loc[split_dict['validations'][val_fold_id], feats + [feat_trgt]]

# Optuna training

## Models setup

seed_target = 1337  # 1337 42 451 1984 1899 1408

models_runs = {
    # 'GANDALF': {
    #     'config': GANDALFConfig,
    #     'n_trials': 1024,
    #     'seed': seed_target,
    #     'n_startup_trials': 256,
    #     'n_ei_candidates': 16
    # },
    # 'FTTransformer': {
    #     'config': FTTransformerConfig,
    #     'n_trials': 1024,
    #     'seed': seed_target,
    #     'n_startup_trials': 256,
    #     'n_ei_candidates': 16
    # },
    'DANet': {
        'config': DANetConfig,
        'n_trials': 512,
        'seed': seed_target,
        'n_startup_trials': 128,
        'n_ei_candidates': 16
    },
    # 'CategoryEmbeddingModel': {
    #     'config': CategoryEmbeddingModelConfig,
    #     'n_trials': 256,
    #     'seed': seed_target,
    #     'n_startup_trials': 64,
    #     'n_ei_candidates': 16
    # },
    # 'TabNetModel': {
    #     'config': TabNetModelConfig,
    #     'n_trials': 256,
    #     'seed': seed_target,
    #     'n_startup_trials': 64,
    #     'n_ei_candidates': 16
    # }
}

## Training

dfs_models = []

for model_name, model_run in models_runs.items():

    model_config_name = model_run['config']
    n_trials = model_run['n_trials']
    seed = model_run['seed']
    n_startup_trials = model_run['n_startup_trials']
    n_ei_candidates = model_run['n_ei_candidates']

    data_config = read_parse_config(f"{path_configs}/DataConfig.yaml", DataConfig)
    data_config['target'] = [feat_trgt]
    data_config['continuous_cols'] = [str(x) for x in feats_cnt]
    data_config['categorical_cols'] = feats_cat
    trainer_config = read_parse_config(f"{path_configs}/TrainerConfig.yaml", TrainerConfig)
    trainer_config['checkpoints_path'] = path_ckpts
    optimizer_config = read_parse_config(f"{path_configs}/OptimizerConfig.yaml", OptimizerConfig)

    lr_find_min_lr = 1e-8
    lr_find_max_lr = 10
    lr_find_num_training = 256
    lr_find_mode = "exponential"
    lr_find_early_stop_threshold = 8.0

    trainer_config['seed'] = seed
    trainer_config['checkpoints'] = 'valid_loss'
    trainer_config['load_best'] = True
    trainer_config['auto_lr_find'] = False

    model_config_default = read_parse_config(f"{path_configs}/models/{model_name}Config.yaml", model_config_name)
    tabular_model_default = TabularModel(
        data_config=data_config,
        model_config=model_config_default,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        verbose=False,
    )
    datamodule = tabular_model_default.prepare_dataloader(train=train, validation=validation, seed=seed)

    opt_parts = ['test', 'validation']
    opt_metrics = [('mean_absolute_error', 'minimize')]
    # opt_metrics = [('mean_absolute_error', 'minimize'), ('pearson_corrcoef', 'maximize')]
    # opt_metrics = [('pearson_corrcoef', 'maximize')]
    opt_directions = []
    for part in opt_parts:
        for metric_pair in opt_metrics:
            opt_directions.append(f"{metric_pair[1]}")

    trials_results = []

    study = optuna.create_study(
        study_name=model_name,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            seed=seed,
        ),
        directions=opt_directions
    )
    study.optimize(
        func=lambda trial: train_hyper_opt(
            trial=trial,
            trials_results=trials_results,
            opt_metrics=opt_metrics,
            opt_parts=opt_parts,
            model_config_default=model_config_default,
            data_config_default=data_config,
            optimizer_config_default=optimizer_config,
            trainer_config_default=trainer_config,
            experiment_config_default=None,
            train=train,
            validation=validation,
            test=test,
            datamodule=datamodule,
            min_lr=lr_find_min_lr,
            max_lr=lr_find_max_lr,
            num_training=lr_find_num_training,
            mode=lr_find_mode,
            early_stop_threshold=lr_find_early_stop_threshold
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )

    fn_trials = (
        f"model({model_name})_"
        f"trials({n_trials}_{seed}_{n_startup_trials}_{n_ei_candidates})_"
        f"tst({tst_split_id})_"
        f"val({val_fold_id})"
    )

    df_trials = pd.DataFrame(trials_results)
    df_trials['split_id'] = tst_split_id
    df_trials['fold_id'] = val_fold_id
    df_trials["train_more"] = False
    df_trials.loc[(df_trials["train_loss"] > df_trials["test_loss"]) | (
            df_trials["train_loss"] > df_trials["validation_loss"]), "train_more"] = True
    df_trials["validation_test_mean_loss"] = (df_trials["validation_loss"] + df_trials["test_loss"]) / 2.0
    df_trials["train_validation_test_mean_loss"] = (df_trials["train_loss"] + df_trials["validation_loss"] + df_trials["test_loss"]) / 3.0
    df_trials.sort_values(by=['test_loss'], ascending=[True], inplace=True)
    df_trials.style.background_gradient(
        subset=[
            "train_loss",
            "validation_loss",
            "validation_test_mean_loss",
            "train_validation_test_mean_loss",
            "test_loss",
            "time_taken",
            "time_taken_per_epoch"
        ], cmap="RdYlGn_r"
    ).to_excel(f"{trainer_config['checkpoints_path']}/{fn_trials}.xlsx")

    dfs_models.append(df_trials)

df_models = pd.concat(dfs_models, ignore_index=True)
df_models.insert(0, 'Selected', 0)
fn = (
    f"models_"
    f"tst({tst_split_id})_"
    f"val({val_fold_id})"
)
df_models.style.background_gradient(
    subset=[
        "train_loss",
        "validation_loss",
        "validation_test_mean_loss",
        "train_validation_test_mean_loss",
        "test_loss",
        "time_taken",
        "time_taken_per_epoch"
    ], cmap="RdYlGn_r"
).to_excel(f"{path_ckpts}/{fn}.xlsx")
