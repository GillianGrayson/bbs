import copy
import time
from typing import Iterable, List, Tuple, Dict
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from scipy import stats
import pickle
import optuna
import pathlib
import lightgbm
import hashlib
import json

      
def get_elastic_net_params_trial(
    trial: optuna.Trial,
    config_default: Dict,
):
    config = copy.deepcopy(config_default)
    config['alpha'] = trial.suggest_float('elastic_net_alpha', 0.00001, 10000, log=True)
    return config

def get_lightgbm_params_trial(
    trial: optuna.Trial,
    config_default: Dict,
):
    config = copy.deepcopy(config_default)
    config['learning_rate'] = trial.suggest_float('lightgbm_learning_rate', 0.005, 5, log=True)
    config['num_leaves'] = trial.suggest_int('lightgbm_num_leaves', 8, 128, step=4)
    config['min_data_in_leaf'] = trial.suggest_int('lightgbm_min_data_in_leaf', 4, 32, step=4)
    config['feature_fraction'] = trial.suggest_float('lightgbm_feature_fraction', 0.5, 1.0)
    config['bagging_fraction'] = trial.suggest_float('lightgbm_bagging_fraction', 0.5, 1.0)
    config['bagging_freq'] = trial.suggest_categorical('lightgbm_bagging_freq', [0, 1, 2, 5, 10, 20, 25, 30])
    return config


def train_hyper_opt_sa_regression(
        trial: optuna.Trial,
        trials_results: List[dict],
        opt_metrics: List[Tuple[str, str]],
        opt_parts: List[str],
        model_config_default: Dict,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        features: Iterable[str],
        target: str,
        save_dir: str,
        verbose: bool = False,
        **kwargs,
):
    """Trains the model with hyperparameter selection from Optuna trials.

    Args:

        trial (optuna.Trial):
            Optuna trial object, which varies hyperparameters.

        trials_results (List[dict]):
            List with results of optuna trials.

        opt_metrics (List[Tuple[str, str]]):
            List of pairs ('metric name', 'direction') for optimization.

        opt_parts (List[str]):
            List of optimization parts: 'train', 'validation', 'test'.

        model_config_default (Dict):
            model_config.

        train (pd.DataFrame): The training data.

        validation (pd.DataFrame): The validation data while training.
            Used in Early Stopping and Logging.

        test (pd.DataFrame): The test data on which performance is evaluated.

        verbose (bool, optional): If True, will print the progress.

        **kwargs: Additional keyword arguments to be passed to the TabularModel fit.

    Returns:
        List of results
    """
    
    start_time = time.time()
    
    data = {
        'train': {'X': train[features].values, 'y': train[target].values},
        'validation': {'X': validation[features].values, 'y': validation[target].values},
        'test': {'X': test[features].values, 'y': test[target].values},
    }

    if model_config_default['name'] == 'elastic_net':
        
        params = get_elastic_net_params_trial(trial, model_config_default)
        check_sum = hashlib.md5(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()
    
        model = ElasticNet(
            alpha=params['alpha'],
            l1_ratio=params['l1_ratio'],
            max_iter=params['max_iter'],
            tol=params['tol'],
        )
        
        model.fit(data['train']['X'], data["train"]["y"])
        
        for part in data:
            data[part]["y_pred"] = model.predict(data[part]['X'])
        
        pathlib.Path(f"{save_dir}/elastic_net_{check_sum}").mkdir(parents=True, exist_ok=True)
        pickle.dump(model, open(f"{save_dir}/elastic_net_{check_sum}/model.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        df_coeffs = pd.Series([model.intercept_] + list(model.coef_), index=['Intercept'] + features, name='coeffs').to_frame()
        df_coeffs.to_excel(f"{save_dir}/elastic_net_{check_sum}/coeffs.xlsx")
        
    elif model_config_default['name'] == 'lightgbm':
        
        params = get_lightgbm_params_trial(trial, model_config_default)
        check_sum = hashlib.md5(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()
        
        ds_trn = lightgbm.Dataset(data['train']['X'], label=data["train"]["y"], feature_name=features)
        ds_val = lightgbm.Dataset(data['train']['X'], label=data["train"]["y"], reference=ds_trn, feature_name=features)
        
        evals_result = {}
        model = lightgbm.train(
            params=params,
            train_set=ds_trn,
            num_boost_round=params.max_epochs,
            valid_sets=[ds_val, ds_trn],
            valid_names=['val', 'train'],
            evals_result=evals_result,
            early_stopping_rounds=params.patience,
            verbose_eval=verbose
        )
        
        for part in data:
            data[part]["y_pred"] = model.predict(data[part]['X'], num_iteration=model.best_iteration)
            
        loss_info = pd.DataFrame(columns=['epoch', 'trn/loss', 'val/loss'])
        loss_info['epoch'] = list(range(len(evals_result['train'][params.metric])))
        loss_info['trn/loss'] = evals_result['train'][params.metric]
        loss_info['val/loss'] = evals_result['val'][params.metric]
        loss_info.to_excel(f"{save_dir}/lightgbm_{check_sum}/loss.xlsx")
        
        model.save_model(f"{save_dir}/lightgbm_{check_sum}/model.model", num_iteration=model.best_iteration)
    
    res_dict = {}   
    for part in data:
        res_dict[f"{part}_mean_absolute_error"] = mean_absolute_error(data[part]["y"], data[part]["y_pred"])
        res_dict[f"{part}_pearson_corrcoef"] = stats.pearsonr(data[part]["y"], data[part]["y_pred"]).statistic
    
    res_dict = {}
    res_dict["time_taken"] = time.time() - start_time
    
    for p in params:
        res_dict[p] = params[p]
    
    trials_results.append(res_dict)
    try:
        pd.DataFrame(trials_results).to_excel(f"{save_dir}/progress.xlsx")
    except PermissionError:
        pass
    
    result = []
    for part in opt_parts:
        for metric_pair in opt_metrics:
            result.append(res_dict[f"{part}_{metric_pair[0]}"])

    return result
