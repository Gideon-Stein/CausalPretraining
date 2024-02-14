import pandas as pd
from os import listdir
import numpy as np
from os import listdir
import yaml



def check_slurm_run(p = "multirun/2023-10-30/14-22-33/", expect_n_runs=False, verbose=0):

    onlyfiles = [ p + f for f in listdir(p)]
    if expect_n_runs:
        assert len(onlyfiles) == expect_n_runs, "seems like there are runs missing: " + str(len(onlyfiles))
    stack = []
    broke = []
    for run in onlyfiles:
        if run == ".submitit":
            pass
        else:
            try:
                yaml_check = [f for f in listdir(run + "/.hydra")]
            except:
                if verbose:
                    print("No Hydra logs")
                continue

            if "overrides.yaml" not in yaml_check:
                if verbose:
                    print("overwrite missing")
                broke.append(run)
                continue
            if "config.yaml" not in yaml_check:
                if verbose:
                    print("config missing") 
                broke.append(run)
                continue
            if "hydra.yaml" not in yaml_check:
                if verbose:
                    print("hydra missing") 
                broke.append(run)
                continue

            
            run_check = [f for f in listdir(run + "/run")]

            if "final_scoring.csv" not in run_check:
                if verbose:
                    print("final_scoring missing") 
                broke.append(run)
                continue
            if "last.ckpt" not in run_check:
                if verbose:
                    print("last missing") 
                broke.append(run)
                continue
            if  not np.any(["-step=" in x for x in run_check]):
                if verbose:
                    print("best missing") 
                broke.append(run)
                continue
            

            with open(run + "/.hydra/overrides.yaml" , 'r') as file:
                meta = yaml.safe_load(file)
                data = [x.split("=") for x in meta]
                meta = pd.DataFrame(data, columns = ["HP", int(run.split("/")[-1])])
                meta.index = meta["HP"]
                meta.drop(columns=["HP"], inplace=True)
                stack.append(meta)
    try:
        stack = pd.concat(stack, axis=1)
    except: 
        stack = None

    return stack, broke


def test_completeness(
    hp_opt,
    relevant=[
        "model.weight_decay",
        "model.optimizer_lr",
        "data.batch_size",
        "corr_input",
        "model.model_type",
        "model.corr_regularization",
        "regression_head",
    ],
):
    if not isinstance(hp_opt, pd.DataFrame):      
        return None
    else:
        count = []
        for x in hp_opt.loc[relevant[0]].unique():
            for y in hp_opt.loc[relevant[1]].unique():
                for z in hp_opt.loc[relevant[2]].unique():
                    for w in hp_opt.loc[relevant[3]].unique():
                        for m in hp_opt.loc[relevant[4]].unique():
                            for n in hp_opt.loc[relevant[5]].unique():
                                for k in hp_opt.loc[relevant[6]].unique():
                                    c1 = hp_opt.loc[relevant[0]] == x
                                    c2 = hp_opt.loc[relevant[1]] == y
                                    c3 = hp_opt.loc[relevant[2]] == z
                                    c4 = hp_opt.loc[relevant[3]] == w
                                    c5 = hp_opt.loc[relevant[4]] == m
                                    c6 = hp_opt.loc[relevant[5]] == n
                                    c7 = hp_opt.loc[relevant[6]] == k
                                    if sum(c1 & c2 & c3 & c4 & c5 & c6 & c7) == 0:
                                        count.append([x, y, z, w, m, n, k])
        count = pd.DataFrame(count, columns = relevant)
        return count