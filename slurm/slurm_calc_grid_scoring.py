import argparse
from model.model_wrapper import Architecture_PL
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
import hydra
import torch
from omegaconf import DictConfig
from pathlib import Path
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from helpers.tools import draw_nx, to_nx_graph
import pickle
import yaml 
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.callbacks import LearningRateMonitor,RichModelSummary,RichProgressBar
import networkx as nx
import pandas as pd
import einops
from matplotlib.pyplot import cm
from helpers.tools import *
from helpers.tools import binary_metrics, lagged_batch_corr, custom_corr_regularization
from data.generator import load_deterministic_ds
from os import listdir
from os.path import isfile, join, isdir
from pathlib import Path
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from helpers.slurm_tools import check_slurm_run, test_completeness
# Loads all checkpoints in a given folder structure and calculates all necessary predictions. Also saves the according labels to be save.
import gc

def load_components(path= "Y"):
    with initialize(version_base=None, config_path=str(path)):
        cfg = compose(config_name="config.yaml")
        cfg.data.batch_size = 16
        model: LightningModule = hydra.utils.instantiate(cfg.model)
        model_path = list(Path(path.parent).rglob("*.ckpt"))
        # select the best not the last checkpoint
        model_path = [x for x in model_path if "step" in str(x)][0]
        model.load_state_dict(torch.load(model_path, map_location="cpu")["state_dict"])
        data_module: LightningDataModule = hydra.utils.instantiate(cfg.data)
        data_module.setup(0)
        # Callbacks
    return model, data_module, cfg


def predict_batch(M,sample, outs, labs, return_regression=False, return_param_prediction=False):

    #check if sample is a list
    if isinstance(sample[0], list):
        sample[0] = [sample[0][0].detach(), sample[0][1].detach()]
    else:
        sample[0] = sample[0].detach()
    # exp 1.1
    if return_regression: 
        print("return reg")
        pred = torch.sigmoid(M.model(sample[0])[1])   
        outs.append(pred.detach())
        labs.append(sample[1].detach())

    #exp 1.3
    elif return_param_prediction:
        print("return params")
        if M.regression_head: 
            pred = M.model(sample[0])[0]
        else:
            pred = M.model(sample[0])
        outs.append(pred.detach())
        if M.regression_head: 
            labs.append(sample[1][0].detach())
        else:
            labs.append(sample[1].detach())

    else:
        if M.regression_head: 
            pred = torch.sigmoid(M.model(sample[0])[0])
        else:
            pred = torch.sigmoid(M.model(sample[0]))
        outs.append(pred.detach())
        if M.regression_head: 
            labs.append(sample[1][0].detach())
        else:
            labs.append(sample[1].detach())
    return outs, labs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="multirun/experiment_1_grid_search/")
    parser.add_argument("--subselect", default= -1)
    parser.add_argument("--skip_finished", action="store_true")


    args = parser.parse_args()

    roc = torchmetrics.classification.BinaryROC()
    auroc = torchmetrics.classification.BinaryAUROC()
    mse = MeanSquaredError()

    mypath = args.data_path
    # experiment folders
    onlyfiles = [ mypath + f for f in listdir(mypath) if not isfile(join(mypath, f)) ]
    if int(args.subselect) != -1:
        onlyfiles = [onlyfiles[int(args.subselect)]]

    # run through all the experiments and create csv summary + error stats
    print(onlyfiles)
    for s in onlyfiles:
        if isfile(s + "_summary.csv"):
            if args.skip_finished:
                print("skipping ", s)
                continue 
        path = Path(s)
        print(s)
        stats, broke = check_slurm_run(s+ "/", verbose=0)
        broken = test_completeness(stats)


        auroc_scoring = []
        mse_scoring = []
        if  isinstance(stats, pd.DataFrame):      
            for run in stats.columns: 
                print(run)
                run_p = Path(s) / str(run) / ".hydra/"
                m, d, cfg = load_components(run_p)
                outs = []
                labs = []
                for p in m.parameters():
                    p.detach()
                # score all the samples.
                for sample in d.val_dataloader():
                    outs, labs = predict_batch(m,sample, outs, labs, return_regression=(cfg.distinguish_mode),
                                                return_param_prediction=(cfg.model.full_representation_mode))
                labs = torch.concat(labs)
                outs = torch.concat(outs)
                auroc_scoring.append(auroc(preds =  outs, target =  labs.type(torch.int64)).detach().cpu().numpy())
                mse_scoring.append(mse(preds =  outs, target =  labs.type(torch.int64)).detach().cpu().numpy())

                del m,d, cfg
                gc.collect()
            stats.loc["Val_Auroc"] = auroc_scoring
            stats.loc["Val_MSE"] = mse_scoring


            stats.to_csv(path.parent / (path.stem + "_summary.csv"))
            broken.to_csv(path.parent / (path.stem + "_broken.csv"))

if __name__ == '__main__':
    main()


