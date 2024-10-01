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
import pathlib
from tqdm import tqdm
import distinctipy
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from statannotations.Annotator import Annotator
from scipy.stats import mannwhitneyu
import shutup

shutup.please()


feats_set = 'inbody'

path = f"D:/YandexDisk/Work/bbd/atlas/subset_{feats_set}"

tst_n_splits = 5
tst_n_repeats = 5
tst_random_state = 1337

val_n_splits = 4
val_n_repeats = 4
val_random_state = 1337

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
                
path_configs = "D:/Work/bbs/notebooks/atlas/configs"

data_config = read_parse_config(f"{path_configs}/DataConfig.yaml", DataConfig)
data_config['target'] = [feat_trgt]
data_config['continuous_cols'] = [str(x) for x in feats_cnt]
data_config['categorical_cols'] = feats_cat
trainer_config = read_parse_config(f"{path_configs}/TrainerConfig.yaml", TrainerConfig)
trainer_config['checkpoints_path'] = "D:/Work/bbs/notebooks/atlas/pt"
optimizer_config = read_parse_config(f"{path_configs}/OptimizerConfig.yaml", OptimizerConfig)

lr_find_min_lr = 1e-8
lr_find_max_lr = 1
lr_find_num_training = 256
lr_find_mode = "exponential"
lr_find_early_stop_threshold = 4.0

search_space = {
    "model_config__gflu_stages": [4, 6],
    "model_config__gflu_dropout": [0.1],
    "model_config__gflu_feature_init_sparsity": [0.2, 0.3],
    "model_config.head_config__dropout": [0.1],
    "model_config__learning_rate": [0.001],
    "model_config__seed": [451],
}
grid_size = np.prod([len(p_vals) for _, p_vals in search_space.items()])
print(grid_size)

head_config = LinearHeadConfig(
    layers="",
    activation='ReLU',
    dropout=0.1,
    use_batch_norm=False,
    initialization="kaiming"
).__dict__

model_list = []
for i, params in enumerate(ParameterGrid(search_space)):
    head_config_tmp = copy.deepcopy(head_config)
    head_config_tmp['dropout'] = params['model_config.head_config__dropout']
    model_config = read_parse_config(f"{path_configs}/models/GANDALFConfig.yaml", GANDALFConfig)
    model_config['gflu_stages'] = params['model_config__gflu_stages']
    model_config['gflu_feature_init_sparsity'] = params['model_config__gflu_feature_init_sparsity']
    model_config['gflu_dropout'] = params['model_config__gflu_dropout']
    model_config['learning_rate'] = params['model_config__learning_rate']
    model_config['seed'] = params['model_config__seed']
    model_config['head_config'] = head_config_tmp
    model_list.append(GANDALFConfig(**model_config))

common_params = {
    "task": "regression",
}

seed = 1408

dfs_result = []
for split_id, split_dict in samples.items():
    for fold_id in split_dict['trains']:
        test = data.loc[split_dict['test'], feats + [feat_trgt]]
        train = data.loc[split_dict['trains'][fold_id], feats + [feat_trgt]]
        validation = data.loc[split_dict['validations'][fold_id], feats + [feat_trgt]]

        trainer_config['seed'] = seed
        trainer_config['checkpoints'] = 'valid_loss'
        trainer_config['load_best'] = True
        trainer_config['auto_lr_find'] = True
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sweep_df, best_model = model_sweep_custom(
                task="regression",
                train=train,
                validation=validation,
                test=test,
                data_config=data_config,
                optimizer_config=optimizer_config,
                trainer_config=trainer_config,
                model_list=model_list,
                common_model_args=common_params,
                metrics=["mean_absolute_error", "pearson_corrcoef"],
                metrics_params=[{}, {}],
                metrics_prob_input=[False, False],
                rank_metric=("mean_absolute_error", "lower_is_better"),
                return_best_model=True,
                seed=seed,
                progress_bar=False,
                verbose=False,
                suppress_lightning_logger=True,
                min_lr = lr_find_min_lr,
                max_lr = lr_find_max_lr,
                num_training = lr_find_num_training,
                mode = lr_find_mode,
                early_stop_threshold = lr_find_early_stop_threshold,
            )
        sweep_df['seed'] = seed
        sweep_df['split_id'] = split_id
        sweep_df['fold_id'] = fold_id
        sweep_df["train_more"] = False
        sweep_df.loc[(sweep_df["train_loss"] > sweep_df["test_loss"]) | (sweep_df["train_loss"] > sweep_df["validation_loss"]), "train_more"] = True
        sweep_df["validation_test_mean_loss"] = (sweep_df["validation_loss"] + sweep_df["test_loss"]) / 2.0
        sweep_df["train_validation_test_mean_loss"] = (sweep_df["train_loss"] + sweep_df["validation_loss"] + sweep_df["test_loss"]) / 3.0
        
        dfs_result.append(sweep_df)
        
        fn_suffix = (f"models({len(model_list)})_"
                     f"tst({tst_random_state}_{tst_n_splits}_{tst_n_repeats})_"
                     f"val({val_random_state}_{val_n_splits}_{val_n_repeats})")
        try:
            df_result = pd.concat(dfs_result, ignore_index=True)
            df_result.sort_values(by=['test_loss'], ascending=[True], inplace=True)
            df_result.style.background_gradient(
                subset=[
                    "train_loss",
                    "validation_loss",
                    "test_loss",
                    "time_taken",
                    "time_taken_per_epoch"
                ], cmap="RdYlGn_r"
            ).to_excel(f"{trainer_config['checkpoints_path']}/{fn_suffix}.xlsx")
        except PermissionError:
            pass