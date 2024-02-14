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


# Similar to the grid scoring, however the folder structure is a little different and we c


def load_components(path= "Y"):
    with initialize(version_base=None, config_path=str(path)):
        cfg = compose(config_name="config.yaml")
        cfg.data.batch_size = 50
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
    parser.add_argument("--skip_finished", action="store_true")
    parser.add_argument("--skip_test_set_2", action="store_true")


    args = parser.parse_args()

    roc = torchmetrics.classification.BinaryROC()
    auroc = torchmetrics.classification.BinaryAUROC()
    mse = MeanSquaredError()

    mypath = args.data_path
    # experiment folders
    onlyfiles = [ mypath + "/"  + f for f in listdir(mypath) if not isfile(join(mypath, f)) ]

    print(onlyfiles)

    result_tables = []

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
        print(len(broken))
    
        mse_scoring_val = []
        auroc_scoring_val = []

        mse_scoring_train = []
        auroc_scoring_train = []

        mse_scoring_test1 = []
        auroc_scoring_test1 = []

        mse_scoring_test2 = []
        auroc_scoring_test2 = []


        for run in stats.columns: 
            print(run)
            run_p = Path(s) / str(run) / ".hydra/"
            m, d, cfg = load_components(run_p)
            
            # #train
            # outs = []
            # labs = []
            # # score all the samples.
            # for sample in d.train_dataloader():
            #     outs, labs = predict_batch(m,sample, outs, labs, return_regression=(cfg.distinguish_mode),
            #                                 return_param_prediction=(cfg.model.full_representation_mode))
            # labs = torch.concat(labs)
            # outs = torch.concat(outs)
            # auroc_scoring_train.append(auroc(preds =  outs, target =  labs.type(torch.int64)).detach().cpu().numpy())
            # mse_scoring_train.append(mse(preds =  outs, target =  labs.type(torch.int64)).detach().cpu().numpy())

            # #val
            # outs = []
            # labs = []
            # # score all the samples.
            # for sample in d.val_dataloader():
            #     outs, labs = predict_batch(m,sample, outs, labs, return_regression=(cfg.distinguish_mode),
            #                                 return_param_prediction=(cfg.model.full_representation_mode))
            # labs = torch.concat(labs)
            # outs = torch.concat(outs)
            # auroc_scoring_val.append(auroc(preds =  outs, target =  labs.type(torch.int64)).detach().cpu().numpy())
            # mse_scoring_val.append(mse(preds =  outs, target =  labs.type(torch.int64)).detach().cpu().numpy())

            #test
            outs = []
            labs = []
            # score all the samples.
            for sample in d.test_dataloader():
                outs, labs = predict_batch(m,sample, outs, labs, return_regression=(cfg.distinguish_mode),
                                            return_param_prediction=(cfg.model.full_representation_mode))
            labs = torch.concat(labs)
            outs = torch.concat(outs)
            auroc_scoring_test1.append(auroc(preds =  outs, target =  labs.type(torch.int64)).detach().cpu().numpy())
            mse_scoring_test1.append(mse(preds =  outs, target =  labs.type(torch.int64)).detach().cpu().numpy())


            #test2
            #reload additional test data

            if not args.skip_test_set_2:
                outs = []
                labs = []
                _, _, data = load_deterministic_ds(ds_name= cfg.data.ds_name +
                                                    "_additional", corr_input=cfg.data.corr_input,
                                                    regression=cfg.regression_head, binary=cfg.data.binary)
                for sample in data:
                    outs, labs = predict_batch(m,sample, outs, labs, return_regression=(cfg.distinguish_mode),
                                                return_param_prediction=(cfg.model.full_representation_mode))
                outs = torch.concat(outs)
                labs = torch.concat(labs)
                auroc_scoring_test2.append(auroc(preds =  outs, target =  labs.type(torch.int64)).detach().cpu().numpy())
                mse_scoring_test2.append(mse(preds =  outs, target =  labs.type(torch.int64)).detach().cpu().numpy())



        # stats.loc["Val_Auroc"] = auroc_scoring_val
        # stats.loc["Val_MSE"] = mse_scoring_val

        # stats.loc["Train_Auroc"] = auroc_scoring_train
        # stats.loc["Train_MSE"] = mse_scoring_train

        stats.loc["Test1_Auroc"] = auroc_scoring_test1
        stats.loc["Test1_MSE"] = mse_scoring_test1

        if not args.skip_test_set_2:
            stats.loc["Test2_AUROC"] = auroc_scoring_test2
            stats.loc["Test2_MSE"] = mse_scoring_test2

        result_tables.append(stats)

    pd.concat(result_tables, axis=1).to_csv(args.data_path + "/summary.csv")



if __name__ == '__main__':
    main()


