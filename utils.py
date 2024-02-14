import warnings
from typing import Callable
import hydra
from lightning.pytorch import Callback
from omegaconf import DictConfig
import torch
from lightning.pytorch.utilities import rank_zero_only
import pandas as pd
import numpy as np
import random

def test_dict_to_csv(metrics: dict, cfg: DictConfig):
    for x in metrics: 
        metrics[x] = metrics[x].detach().numpy()
    pd.DataFrame.from_dict(
        data=metrics, orient='index').to_csv(cfg.tensorboard.save_dir + "/" + 'final_scoring.csv', header=False)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """
    if not cfg.get("extras"):
        return

    # if cfg.get("tensorboard"):
    #     exp_name = cfg.get("name", None)
    #     date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     ds_name = cfg.data._target_.split(".")[-1].replace("_DataModule", "")
    #     mo_name = cfg.model._target_.split(".")[-1].replace("Module", "")
    #     if exp_name:
    #         cfg.tensorboard.save_dir = cfg.tensorboard.save_dir + "/" + exp_name
        
    #     cfg.tensorboard.save_dir = cfg.tensorboard.save_dir + "/" + ds_name + "/" + date + "/" + mo_name

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        warnings.filterwarnings("ignore")
        
    if cfg.extras.get("pretty_errors"):
        try:
            import pretty_errors
            pretty_errors.configure(separator_character = '*')
        except ImportError:
            pass

    if not cfg.trainer.get("enable_checkpointing"):
        cfg.trainer.enable_checkpointing = False
        print("Checkpointing disabled!") # TODO Make it a log and red :^)

def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
           pass

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        # log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    # log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

    
@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.
    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

        
def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config."""
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    if not callbacks_cfg:
        return []

    callbacks: list[Callback] = []
    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def select_least_tortured_gpu(verbose=False):
    """
    figure out which gpu runs the least processes.
    """
    available_devices = list(range(torch.cuda.device_count()))
    device_names = [torch.cuda.get_device_name(d) for d in available_devices]

    if verbose:
        print(f"found {len(device_names)} devices:")
        print(device_names)

    if len(available_devices) == 1:
        return available_devices

    # you need to install pynvml -> pip install pynvml

    processes = []
    for device in available_devices:
        processes.append(torch.cuda.list_gpu_processes(torch.device(device)))

    if verbose:
        for p in processes:
            print(p)

    p_counts = [p.count("memory") for p in processes]
    min_indices = list(np.where(p_counts == np.min(p_counts))[0])
    random.shuffle(min_indices)
    if verbose:
        print(f"Selected GPU {available_devices[min_indices[0]]}")
    return [available_devices[min_indices[0]]]