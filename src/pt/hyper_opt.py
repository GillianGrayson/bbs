import copy
import time
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rich.progress import Progress, track

from pytorch_tabular import TabularModel, models
from pytorch_tabular.models import (
    CategoryEmbeddingModelConfig,
    GANDALFConfig,
    TabNetModelConfig,
    FTTransformerConfig,
    DANetConfig
)
from pytorch_tabular.tabular_datamodule import TabularDatamodule
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.utils import (
    OOMException,
    OutOfMemoryHandler,
    available_models,
    get_logger,
    int_to_human_readable,
    suppress_lightning_logs,
)
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner

import optuna

from src.utils.configs import read_parse_config

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

logger = get_logger("pytorch_tabular")


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield err, out


def get_model_config_trial(
        trial: optuna.Trial,
        model_config_default
):
    params_head = LinearHeadConfig(
        layers="",
        activation='ReLU',
        dropout=trial.suggest_float('head_dropout', 0.0, 0.2),
        use_batch_norm=False,
        initialization="kaiming"
    ).__dict__

    if model_config_default._model_name == 'GANDALFModel':
        model_config = copy.deepcopy(model_config_default)
        model_config['gflu_stages'] = trial.suggest_int('gflu_stages', 1, 15)
        model_config['gflu_dropout'] = trial.suggest_float('gflu_dropout', 0.0, 0.2)
        model_config['gflu_feature_init_sparsity'] = trial.suggest_float('gflu_feature_init_sparsity', 0.05, 0.55)
        model_config['learning_rate'] = 0.001
        model_config['seed'] = 1337
        model_config['head_config'] = params_head
    else:
        raise ValueError(f"Model {model_config_default._model_name} not supported for Optuna trials")

    return model_config


def train_hyper_opt(
        trial: optuna.Trial,
        trials_results: List[dict],
        opt_metrics: List[Tuple[str, str]],
        opt_parts: List[str],
        model_config_default: Union[ModelConfig, str],
        data_config: Union[DataConfig, str],
        optimizer_config: Union[OptimizerConfig, str],
        trainer_config: Union[TrainerConfig, str],
        experiment_config: Optional[Union[ExperimentConfig, str]],
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        datamodule: TabularDatamodule,
        min_lr: float = 1e-8,
        max_lr: float = 1,
        num_training: int = 100,
        mode: str = "exponential",
        early_stop_threshold: Optional[float] = 4.0,
        handle_oom: bool = True,
        ignore_oom: bool = True,
        verbose: bool = False,
        suppress_lightning_logger: bool = True,
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

        model_config_default (Union[ModelConfig, str]):
            A subclass of ModelConfig or path to the yaml file with default model configuration.
            Determines which model to run from the type of config.

        data_config (Union[DataConfig, str]):
            DataConfig object or path to the yaml file. Defaults to None.

        optimizer_config (Union[OptimizerConfig, str]): The OptimizerConfig for the TabularModel.
            If str is passed, will initialize the OptimizerConfig using the yaml file in that path.

        trainer_config (Union[TrainerConfig, str]): The TrainerConfig for the TabularModel.
            If str is passed, will initialize the TrainerConfig using the yaml file in that path.

        experiment_config (Union[ExperimentConfig, str]): ExperimentConfig object or path to the yaml file.

        train (pd.DataFrame): The training data.

        validation (pd.DataFrame): The validation data while training.
            Used in Early Stopping and Logging.

        test (pd.DataFrame): The test data on which performance is evaluated.

        datamodule (TabularDatamodule): The datamodule.

        min_lr (Optional[float], optional): minimum learning rate to investigate

        max_lr (Optional[float], optional): maximum learning rate to investigate

        num_training (Optional[int], optional): number of learning rates to test

        mode (Optional[str], optional): search strategy, either 'linear' or 'exponential'.
            If set to 'linear' the learning rate will be searched by linearly increasing after each batch.
            If set to 'exponential', will increase learning rate exponentially.

        early_stop_threshold (Optional[float], optional): threshold for stopping the search.
            If the loss at any point is larger than early_stop_threshold*best_loss then the search is stopped.
            To disable, set to None.

        handle_oom (bool): If True, will try to handle OOM errors elegantly.

        ignore_oom (bool, optional): If True, will ignore the Out of Memory error and continue with the next model.

        verbose (bool, optional): If True, will print the progress.

        suppress_lightning_logger (bool, optional): If True, will suppress the lightning logger.

        **kwargs: Additional keyword arguments to be passed to the TabularModel fit.

    Returns:
        pl.Trainer: The PyTorch Lightning Trainer instance
    """
    if suppress_lightning_logger:
        suppress_lightning_logs()

    model_config_trial = get_model_config_trial(trial, read_parse_config(model_config_default, ModelConfig))
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config_trial,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        experiment_config=experiment_config,
        verbose=verbose,
        suppress_lightning_logger=suppress_lightning_logger
    )

    prep_dl_kwargs, prep_model_kwargs, train_kwargs = tabular_model._split_kwargs(kwargs)

    start_time = time.time()

    model = tabular_model.prepare_model(datamodule, **prep_model_kwargs)

    tabular_model._prepare_for_training(model, datamodule, **train_kwargs)
    train_loader, val_loader = (
        tabular_model.datamodule.train_dataloader(),
        tabular_model.datamodule.val_dataloader(),
    )
    tabular_model.model.train()
    if tabular_model.config.auto_lr_find and (not tabular_model.config.fast_dev_run):
        if tabular_model.verbose:
            logger.info("Auto LR Find Started")
        with OutOfMemoryHandler(handle_oom=handle_oom) as oom_handler:
            with suppress_stdout_stderr():
                lr_finder = Tuner(tabular_model.trainer).lr_find(
                    tabular_model.model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    min_lr=min_lr,
                    max_lr=max_lr,
                    num_training=num_training,
                    mode=mode,
                    early_stop_threshold=early_stop_threshold,
                )
        if oom_handler.oom_triggered:
            raise OOMException(
                "OOM detected during LR Find. Try reducing your batch_size or the"
                " model parameters." + "/n" + "Original Error: " + oom_handler.oom_msg
            )
        if tabular_model.verbose:
            logger.info(
                f"Suggested LR: {lr_finder.suggestion()}. For plot and detailed"
                " analysis, use `find_learning_rate` method."
            )
        tabular_model.model.reset_weights()
        # Parameters in models needs to be initialized again after LR find
        tabular_model.model.data_aware_initialization(tabular_model.datamodule)

    tabular_model.model.train()
    if tabular_model.verbose:
        logger.info("Training Started")
    with OutOfMemoryHandler(handle_oom=handle_oom) as oom_handler:
        tabular_model.trainer.fit(tabular_model.model, train_loader, val_loader)
    if oom_handler.oom_triggered:
        raise OOMException(
            "OOM detected during Training. Try reducing your batch_size or the"
            " model parameters."
            "/n" + "Original Error: " + oom_handler.oom_msg
        )
    tabular_model._is_fitted = True
    if tabular_model.verbose:
        logger.info("Training the model completed")
    if tabular_model.config.load_best:
        tabular_model.load_best_model()

    res_dict = {
        "model": tabular_model.name,
        'learning_rate': lr_finder.suggestion(),
        "# Params": int_to_human_readable(tabular_model.num_params),
    }
    if oom_handler.oom_triggered:
        if not ignore_oom:
            raise OOMException(
                "Out of memory error occurred during cross validation. "
                "Set ignore_oom=True to ignore this error."
            )
        else:
            res_dict.update(
                {
                    f"test_loss": np.inf,
                    f"validation_loss": np.inf,
                    "epochs": "OOM",
                    "time_taken": "OOM",
                    "time_taken_per_epoch": "OOM",
                }
            )
            for part in opt_parts:
                for metric_pair in opt_metrics:
                    res_dict[f"{part}_{metric_pair[0]}"] = np.inf if metric_pair[1] == 'minimize' else -np.inf
            res_dict["model"] = tabular_model.name + " (OOM)"
    else:
        if (
                tabular_model.trainer.early_stopping_callback is not None
                and tabular_model.trainer.early_stopping_callback.stopped_epoch != 0
        ):
            res_dict["epochs"] = tabular_model.trainer.early_stopping_callback.stopped_epoch
        else:
            res_dict["epochs"] = tabular_model.trainer.max_epochs

        # Update results with train metrics
        train_metrics = tabular_model.evaluate(test=train, verbose=False)[0]
        metrics_names = list(train_metrics.keys())
        for m_name in metrics_names:
            train_metrics[m_name.replace('test', 'train')] = train_metrics.pop(m_name)
        res_dict.update(train_metrics)

        # Update results with validation metrics
        validation_metrics = tabular_model.evaluate(test=validation, verbose=False)[0]
        metrics_names = list(validation_metrics.keys())
        for m_name in metrics_names:
            validation_metrics[m_name.replace('test', 'validation')] = validation_metrics.pop(m_name)
        res_dict.update(validation_metrics)

        # Update results with test metrics
        res_dict.update(tabular_model.evaluate(test=test, verbose=False)[0])

        res_dict["time_taken"] = time.time() - start_time
        res_dict["time_taken_per_epoch"] = res_dict["time_taken"] / res_dict["epochs"]

        if verbose:
            logger.info(f"Finished Training {tabular_model.name}")
            logger.info("Results:" f" {', '.join([f'{k}: {v}' for k, v in res_dict.items()])}")
        res_dict["params"] = model_config_trial

        if tabular_model.trainer.checkpoint_callback:
            res_dict["checkpoint"] = tabular_model.trainer.checkpoint_callback.best_model_path

        trials_results.append(res_dict)

        if tabular_model.config['checkpoints_path']:
            try:
                pd.DataFrame(trials_results).style.background_gradient(
                    subset=[
                        "train_loss",
                        "validation_loss",
                        "test_loss",
                        "time_taken",
                        "time_taken_per_epoch"
                    ], cmap="RdYlGn_r"
                ).to_excel(f"{tabular_model.config['checkpoints_path']}/progress.xlsx")
            except PermissionError:
                pass

    result = []
    for part in opt_parts:
        for metric_pair in opt_metrics:
            result.append(res_dict[f"{part}_{metric_pair[0]}"])

    return result
