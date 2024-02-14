import argparse
import pickle
from helpers.tools import *
import os
import subprocess
import time

# Takes the summary table created by summarize_exp, selects the best run and parses it to a command that is executed n times with different seeds
# Runs the slurm command directly. seeds are treated as the only hyperparameters to the slurm command


def start_single_process(command, server="stein@triton.inf-cv.uni-jena.de"):
    cmd = (
        " . anaconda3/etc/profile.d/conda.sh && conda activate causal_pretraining && cd ../stein_test/Causal_pretraining && "
        + command
    )
    ssh = subprocess.Popen(["ssh", server, cmd], shell=False, stdout=None, stderr=True)
    return ssh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", default="multirun/experiment_1_grid_search/")
    parser.add_argument("--not_run", action="store_true")
    parser.add_argument("--subselection", action="store_true")


    args = parser.parse_args()

    mypath = args.exp_path
    onlyfiles = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted([x for x in onlyfiles if "summary" in x])
    res = pd.concat([pd.read_csv(x, index_col=0) for x in onlyfiles], axis=1).T


    #res = pd.read_csv(mypath + "21-11-12-42-240387_summary.csv", index_col=0).T
    # fix the default values for the model types.
    print(res["model.model_type"].unique())

    res.loc[res["data.ds_name"].isnull(), "data.ds_name"] = "base"
    execute = []

    base = "python train.py -m hparams_search=std_runs_slurm.yaml"
    m_holder = res["model.model_type"].unique()
    d_holder =  res["data.ds_name"].unique()
    s_holder = res["model"].unique()
    # by hand to skip some aready finished runs.
    if args.subselection:
        d_holder = ["scale_up_2"]
        s_holder =  ["small"]
    for model in m_holder:
        for data in d_holder:
            for size in s_holder:
                values = res[
                    (res["model.model_type"] == model)
                    & (res["data.ds_name"] == data)
                    & (res["model"] == size)
                ].sort_values("Val_MSE", ascending=True)[:1]
                if len(values) == 0:
                    continue
                for col in values.columns:
                    print(col)
                    if col == "Val_MSE":
                        continue
                    elif col == "Val_Auroc":
                        continue
                    elif col == "Test1_Auroc":
                        continue
                    elif col == "Test1_MSE":
                        continue
                    elif col == "Test2_AUROC":
                        continue
                    elif col == "Test2_MSE":
                        continue
                    elif col == "hparams_search":
                        continue
                    else:
                        base += " " + col + "=" + values[col].values[0]
                execute.append(base)

    print("Commands to execute: " + str(len(execute)))
    if not args.not_run:
        for command in execute:
            start_single_process(command)
            time.sleep(61)
    else:
        print(execute)


if __name__ == "__main__":
    main()
