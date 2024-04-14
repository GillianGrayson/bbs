from pytorch_tabular.utils import load_covertype_dataset
from rich.pretty import pprint
import torch
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
from pytorch_tabular.models import CategoryEmbeddingModelConfig, GANDALFConfig, TabNetModelConfig, FTTransformerConfig, DANetConfig, GatedAdditiveTreeEnsembleConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.tabular_model_tuner import TabularModelTuner
from torchmetrics.functional.regression import mean_absolute_error, pearson_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold
from pytorch_tabular import MODEL_SWEEP_PRESETS
import pandas as pd
from pytorch_tabular import model_sweep
from src.pt.model_sweep import model_sweep_custom
import warnings
from src.utils.configs import read_parse_config
from src.utils.hash import dict_hash
from pytorch_tabular.utils import get_balanced_sampler

path_data = "D:/YandexDisk/Work/bbd/immunology/002_central_vs_yakutia/classification"
path_configs = "D:/Work/bbs/notebooks/immunology/002_central_vs_yakutia/pt_configs"
data = pd.read_excel(f"{path_data}/data.xlsx", index_col=0)
feats = pd.read_excel(f"{path_data}/feats.xlsx", index_col=0).index.values.tolist()

test_split_id = 0

val_n_splits = 4
val_random_state = 1337
val_fold_id = 0

for fold_id in range(val_n_splits):
    data[f"Fold_{fold_id}"] = data[f"Split_{test_split_id}"]

stratify_cat_parts = {
    'Central': data.index[(data['Region'] == 'Central') & (data[f"Split_{test_split_id}"] == 'trn_val')].values,
    'Yakutia': data.index[(data['Region'] == 'Yakutia') & (data[f"Split_{test_split_id}"] == 'trn_val')].values,
}
for part, ids in stratify_cat_parts.items():
    print(f"{part}: {len(ids)}")
    con = data.loc[ids, 'Age'].values
    ptp = np.ptp(con)
    num_bins = 5
    bins = np.linspace(np.min(con) - 0.1 * ptp, np.max(con) + 0.1 * ptp, num_bins + 1)
    binned = np.digitize(con, bins) - 1
    unique, counts = np.unique(binned, return_counts=True)
    occ = dict(zip(unique, counts))
    k_fold = RepeatedStratifiedKFold(
        n_splits=val_n_splits,
        n_repeats=1,
        random_state=val_random_state
    )
    splits = k_fold.split(X=ids, y=binned, groups=binned)

    for fold_id, (ids_trn, ids_val) in enumerate(splits):
        data.loc[ids[ids_trn], f"Fold_{fold_id}"] = "trn"
        data.loc[ids[ids_val], f"Fold_{fold_id}"] = "val"

test = data.loc[data[f"Split_{test_split_id}"] == "tst", feats + ['Region']]
train_validation = data.loc[
    data[f"Split_{test_split_id}"] == "trn_val", feats + ['Region'] + [f"Fold_{i}" for i in range(val_n_splits)]]
train_only = data.loc[data[f"Fold_{val_fold_id}"] == "trn", feats + ['Region']]
validation_only = data.loc[data[f"Fold_{val_fold_id}"] == "val", feats + ['Region']]
cv_indexes = [
    (
        np.where(train_validation.index.isin(train_validation.index[train_validation[f"Fold_{i}"] == 'trn']))[0],
        np.where(train_validation.index.isin(train_validation.index[train_validation[f"Fold_{i}"] == 'val']))[0],
    )
    for i in range(val_n_splits)
]


sampler_balanced = get_balanced_sampler(train_only['Region'].values.ravel())


data_config = read_parse_config(f"{path_configs}/DataConfig.yaml", DataConfig)
optimizer_config = read_parse_config(f"{path_configs}/OptimizerConfig.yaml", OptimizerConfig)
trainer_config = read_parse_config(f"{path_configs}/TrainerConfig.yaml", TrainerConfig)
trainer_config['seed'] = 1337
trainer_config['checkpoints'] = 'valid_loss'
trainer_config['load_best'] = True
trainer_config['auto_lr_find'] = True
data_config['continuous_feature_transform'] = 'yeo-johnson'  # 'box-cox' 'yeo-johnson' 'quantile_normal'
data_config['normalize_continuous_features'] = True

sweep_df = pd.read_excel(
    f"{trainer_config['checkpoints_path']}/sweep_1337_CosineAnnealingWarmRestarts_yeo-johnson.xlsx", index_col=0)

model_id = 197

tabular_model = TabularModel(
    data_config=data_config,
    model_config=ast.literal_eval(sweep_df.at[model_id, 'params']),
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    verbose=True,
    suppress_lightning_logger=False
)
datamodule = tabular_model.prepare_dataloader(
    train=train_only,
    validation=validation_only,
    seed=1337,
)
model = tabular_model.prepare_model(
    datamodule
)
tabular_model._prepare_for_training(
    model,
    datamodule
)
tabular_model.load_weights(sweep_df.at[model_id, 'checkpoint'])
tabular_model.evaluate(test, verbose=False)
tabular_model.save_model(f"{tabular_model.config['checkpoints_path']}/candidates/{model_id}")

loaded_model = TabularModel.load_model(f"{tabular_model.config['checkpoints_path']}/candidates/{model_id}")

df = data.loc[:, ['Age', 'SImAge', 'Sex', 'Region']]
df.loc[train_only.index, 'Group'] = 'Train'
df.loc[validation_only.index, 'Group'] = 'Validation'
df.loc[test.index, 'Group'] = 'Test'
prediction = loaded_model.predict(data)
df.to_csv(f"{loaded_model.config['checkpoints_path']}/candidates/{model_id}/df.xlsx")