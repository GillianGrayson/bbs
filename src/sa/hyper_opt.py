import time
from typing import Iterable, List, Tuple
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from scipy import stats
import pickle
import optuna
import pathlib

      
def get_elastic_net_params_trial(
    trial: optuna.Trial,
):
    params = {
        'alpha': trial.suggest_float('elastic_net_alpha', 0.00001, 10000, log=True),
        'l1_ratio': 0.5,
        'max_iter': 100000,
        'tol': 1e-2
    }
    return params


def train_hyper_opt_sa(
        trial: optuna.Trial,
        trials_results: List[dict],
        opt_metrics: List[Tuple[str, str]],
        opt_parts: List[str],
        model_type: str,
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

        model (str):
            Type of model.

        train (pd.DataFrame): The training data.

        validation (pd.DataFrame): The validation data while training.
            Used in Early Stopping and Logging.

        test (pd.DataFrame): The test data on which performance is evaluated.

        verbose (bool, optional): If True, will print the progress.

        **kwargs: Additional keyword arguments to be passed to the TabularModel fit.

    Returns:
        List of results
    """

    params = get_elastic_net_params_trial(trial)
    
    model = ElasticNet(
        alpha=params['alpha'],
        l1_ratio=params['l1_ratio'],
        max_iter=params['max_iter'],
        tol=params['tol'],
    )
    
    data = {
        'train': {'X': train[features].values, 'y': train[target].values},
        'validation': {'X': validation[features].values, 'y': validation[target].values},
        'test': {'X': test[features].values, 'y': test[target].values},
    }
    
    start_time = time.time()
    
    model.fit(data['train']['X'], data["train"]["y"])
    
    data["train"]["y_pred"] = model.predict(data['train']['X'])
    data["validation"]["y_pred"] = model.predict(data['validation']['X'])
    data["test"]["y_pred"] = model.predict(data['test']['X'])
    
    res_dict = {
        'alpha': params['alpha'],
        'l1_ratio': params['l1_ratio'],
        'max_iter': params['max_iter'],
        "tol": params['tol'],
    }
    res_dict["time_taken"] = time.time() - start_time
    
    for part in data:
        res_dict[f"{part}_mean_absolute_error"] = mean_absolute_error(data[part]["y"], data[part]["y_pred"])
        res_dict[f"{part}_pearson_corrcoef"] = stats.pearsonr(data[part]["y"], data[part]["y_pred"]).statistic
    
    trials_results.append(res_dict)
    try:
        pd.DataFrame(trials_results).to_excel(f"{save_dir}/progress.xlsx")
    except PermissionError:
        pass
    
    pathlib.Path(f"{save_dir}/elastic_net_{params['alpha']:0.4e}").mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(f"{save_dir}/elastic_net_{params['alpha']:0.4e}/model.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    df_coeffs = pd.Series([model.intercept_] + list(model.coef_), index=['Intercept'] + features, name='coeffs').to_frame()
    df_coeffs.to_excel(f"{save_dir}/elastic_net_{params['alpha']:0.4e}/coeffs.xlsx")

    result = []
    for part in opt_parts:
        for metric_pair in opt_metrics:
            result.append(res_dict[f"{part}_{metric_pair[0]}"])

    return result
