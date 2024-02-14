import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.pyplot import cm
import sys
from yaml import safe_load
import pathlib
import pandas as pd
from os import listdir
from os.path import isfile, join, isdir
import torch.nn as nn
import torchmetrics

sys.path.append("..")
import einops
from statsmodels.tsa.api import VAR
import pickle
from tigramite.pcmci import PCMCI
from pathlib import Path


from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print('took: %2.4f sec' % \
        #  (te-ts))
        return result, te - ts

    return wrap


class weighted_mse:
    # might be worth to add prob sparse attention here but for now lets just use full
    def __init__(self, scaling=90):
        self.mse = nn.MSELoss(reduction="none")
        self.scaling = scaling

    def __call__(self, inp, target):
        # get target weight vector
        weights = torch.ones(target.shape, device=inp.device)
        weights[target > 0] = self.scaling
        return (weights * self.mse(inp, target)).mean()


def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans("".join(normal), "".join(sub_s))
    return x.translate(res)


def to_nx_graph(g, labels=["A", "B", "C", "D", "E", "F", "G"]):
    G = nx.DiGraph()
    # Makes a node for every possible lag
    for n, x in enumerate(labels[: g.shape[0]]):
        for y in range(g.shape[2] + 1):
            if y == 0:
                name = ""
            else:
                name = get_sub("T-" + str(y))  # H₂SO₄

            # G.add_node(x + name,pos=(g.shape[2]+1-y ,g.shape[0]- n))
            G.add_node(x + name, pos=(g.shape[2] + 1 - y, g.shape[0] - n))

    # Adds the edges according to the format, effect, cause, lag
    for n, effect in enumerate(g):
        for m, cause in enumerate(effect):
            for o, lag in enumerate(cause):
                if not lag == 0:
                    G.add_edge(
                        labels[m] + get_sub("T-" + str(g.shape[2] - o)),
                        labels[n],
                        weight=round(float(lag.numpy()), 2),
                    )
    return G


def draw_nx(
    G,
    ax=None,
    title="Generative process",
    n_vars=5,
    n_lags=3,
    node_size=2500,
    colors=None,
    draw_labels=True,
):
    # set color
    c = {}
    node_s = []
    color_count = [
        item for sublist in [[x] * (n_lags + 1) for x in colors] for item in sublist
    ]
    for n, node in enumerate(G.nodes):
        if len(node) == 1:
            node_s.append(node_size * 1.5)
        else:
            node_s.append(node_size)

        c[node] = color_count[n]

    nx.set_node_attributes(G, c, name="color")
    pos = nx.get_node_attributes(G, "pos")
    c = nx.get_node_attributes(G, "color")
    labels = nx.get_edge_attributes(G, "weight")
    if ax:
        pass
    else:
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(1, 1, 1)
    # ax.axhline(y = 1, color = 'lightgrey', linestyle = '-', alpha=0.3)
    nx.draw_networkx(
        G,
        pos,
        node_color=[c[x] for x in c],
        alpha=0.75,
        arrowstyle="->",
        arrows=True,
        with_labels=True,
        # edge_color=colors,
        width=3,
        connectionstyle="Angle3, angleA=70, angleB=0",
        arrowsize=10,
        node_size=node_s,
        font_weight="bold",
        # linewidths=40,
        font_size=10,
        ax=ax,
    )
    if draw_labels:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=labels, ax=ax, label_pos=0.25, rotate=False, font_size=8
        )

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xticks(ticks=[])
    ax.set_yticks(ticks=[])

    # ax.set_ylabel("Variable", fontsize = 15)
    ax.set_xlabel("Timesteps", fontsize=8)
    # ax.set_xlim([1.1*x for x in ax.get_xlim()])
    ax.set_ylim([0, n_vars + 1])
    ax.set_title(title, fontsize=20)

    for x in range(1, n_vars + 1):
        ax.arrow(
            1,
            x,
            n_lags + (0.1 * n_lags),
            0,
            linestyle="dotted",
            alpha=0.3,
            head_width=0.2,
            head_length=0.04,
        )

def display_example( X,pred,Y, filter=None,  draw_labels=False):
    labels = ["A", "B", "C", "D", "E", "F", "G"]

    n_vars = pred.shape[2]

    color = list(iter(cm.Paired(np.linspace(0, 1, n_vars))))
    mosaic = "LP;"
    for ts in range(n_vars):
        mosaic += str(labels[ts]) + str(labels[ts]) + ";"
    hr = [4] + list(np.ones(n_vars).astype(int))
    fig, axs = plt.subplot_mosaic(
        mosaic[:-1], gridspec_kw={"height_ratios": hr}, figsize=(12, 3 + n_vars)
    )

    pred_clas = pred[0].cpu().detach()

    if filter:
        pred_clas[pred_clas < filter] = 0
    for ts in range(n_vars):
        axs[labels[ts]].plot(X[:, ts], color=color[ts])
        axs[labels[ts]].set_ylabel(labels[ts], fontsize=14)

    draw_nx(
        to_nx_graph(Y),
        ax=axs["L"],
        n_vars=Y.shape[0],
        n_lags=Y.shape[2],
        node_size=int(2500 / Y.shape[0]),
        colors=color,
        draw_labels=draw_labels,
    )

    draw_nx(
        to_nx_graph(pred_clas),
        title="Predicted process",
        ax=axs["P"],
        n_vars=Y.shape[0],
        n_lags=Y.shape[2],
        node_size=int(2500 / Y.shape[0]),
        colors=color,
        draw_labels=draw_labels,
    )

    for ts in range(n_vars):
        axs[labels[ts]].get_xaxis().set_visible(False)
        axs[labels[ts]].get_yaxis().set_visible(False)

    axs[labels[n_vars - 1]].set_xlabel("Timesteps")

    plt.show()


def display_training(loss, eval_res):
    plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    # ax1.set_yscale("log")
    ax1.plot(loss["avg_loss"])
    ax2.plot(eval_res["MAE"])
    ax1.set_title("Training loss")
    ax2.set_title("Test MAE after epoch")
    plt.show()


def binary_metrics(pred, lab, link_threshold, p_value=False):
    if not p_value:
        binary = pred > link_threshold
    else:
        binary = pred < link_threshold

    tp = torch.sum((binary == 1) * (lab == 1))
    tn = torch.sum((binary == 0) * (lab == 0))
    fp = torch.sum((binary == 1) * (lab == 0))
    fn = torch.sum((binary == 0) * (lab == 1))
    assert torch.all(tp + fp + tn + fn), "BROKEN metric"
    return tp / (tp + fn), fp / (fp + tn), tn / (fp + tn), fn / (tp + fn)


def lagged_batch_corr(points, max_lags):
    # calculates the autocovariance matrix with a batch dimension
    # lagged variables are concated in the same dimension.
    # inpuz (B, time, var)
    # roll to calculate lagged cov:
    B, N, D = points.size()

    # we roll the data and add it together to have the lagged versions in the table
    stack = torch.concat(
        [torch.roll(points, x, dims=1) for x in range(max_lags + 1)], dim=2
    )

    mean = stack.mean(dim=1).unsqueeze(1)
    std = stack.std(dim=1).unsqueeze(1)
    diffs = (stack - mean).reshape(B * N, D * (max_lags + 1))

    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(
        B, N, D * (max_lags + 1), D * (max_lags + 1)
    )

    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    # make correlation out of it by dividing by the product of the stds
    corr = bcov / (
        std.repeat(1, D * (max_lags + 1), 1).reshape(
            std.shape[0], D * (max_lags + 1), D * (max_lags + 1)
        )
        * std.permute((0, 2, 1))
    )
    # we can remove backwards in time links. (keep only the original values)
    return corr[:, :D, D:]  # (B, D, D)


def transform_corr_to_y(corr, ml, n_vars):
    ncorr = einops.rearrange(corr[:, :n_vars:], "b c1 (t c2) -> b c1 c2 t", t=ml)
    fncorr = torch.flip(ncorr, dims=[3])
    return fncorr


def custom_corr_regularization(predictions, data, exp=1.5, epsilon=0.15):
    # predictions: (batch.caused, causing, lag)
    # data: batch, t, n_vars
    # penalized predictions if the cov of the corresponding link is low.
    ml = predictions.shape[3]
    n_vars = data.shape[2]

    # rashape everything properly
    corr = lagged_batch_corr(data, ml)
    fncorr = transform_corr_to_y(corr, ml, n_vars)
    # specifying the batch size
    regularization = 1 / (torch.abs(fncorr) + epsilon)  # for numeric stability

    # keep this ot for now to make it simple. Wanted to rescale to a specific range.
    # a = regularization.shape[1]
    # b = regularization.shape[2]
    # c = regularization.shape[3]
    # we rescale every process to 0/1
    # maxi = torch.amax(regularization , dim=[1,2,3]).repeat(a*b*c).reshape(
    #    (a*b*c,-1)).T.reshape(-1,a,b,c)
    # mini= torch.amin(regularization , dim=[1,2,3]).repeat(a*b*c).reshape(
    #    (a*b*c,-1)).T.reshape(-1,a,b,c)
    #
    # rescale =   ((regularization - mini)  /
    #                        (maxi- mini))
    penalty = torch.mean((predictions * regularization) ** exp)
    return penalty


def visualize_y(y, labels=True):
    n_vars, _, lags = y.shape
    color = list(iter(cm.Paired(np.linspace(0, 1, n_vars))))

    draw_nx(
        to_nx_graph(y),
        title="Predicted process",
        n_vars=n_vars,
        n_lags=lags,
        node_size=int(2500 / 3),
        draw_labels=labels,
        colors=color,
    )


def test_completeness(params):
    hp_opt = params
    wd = hp_opt.loc["model.weight_decay"].unique()
    opt = hp_opt.loc["model.optimizer_lr"].unique()
    bs = hp_opt.loc["data.batch_size"].unique()
    ci = hp_opt.loc["corr_input"].unique()
    cr = hp_opt.loc["model.corr_regularization"].unique()
    rh = hp_opt.loc["regression_head"].unique()

    count = 0
    completeC = 0
    for x in ci:
        for y in wd:
            for z in opt:
                for w in bs:
                    for m in cr:
                        for n in rh:
                            c1 = hp_opt.loc["corr_input"] == x
                            c2 = hp_opt.loc["model.weight_decay"] == y
                            c3 = hp_opt.loc["model.optimizer_lr"] == z
                            c4 = hp_opt.loc["data.batch_size"] == w
                            c5 = hp_opt.loc["model.corr_regularization"] == m
                            c6 = hp_opt.loc["regression_head"] == n
                            completeC += 1
                            if sum(c1 & c2 & c3 & c4 & c5 & c6) == 0:
                                if m ^ n:
                                    print(x, y, z, w, m, n)
                                    count += 1
    # print(x, y,z,w,m,n, sum( c1 & c2 & c3 & c4 & c5 & c6 ))
    print(count, completeC - 24)


def make_result_grid_table(path):
    mypath = pathlib.Path(pathlib.Path(path) / "GeneratorDataModule/")
    params = []
    folders = [f for f in listdir(mypath) if not isfile(join(mypath, f))]

    for x in folders:
        with open(mypath / x / "Architecture_PL/version_0/hparams.yaml", "r") as f:
            df = pd.json_normalize(safe_load(f))
            df = pd.DataFrame(df.values.T, index=df.columns)
            df.rename(columns={0: x}, inplace=True)
        params.append(df)
    params = pd.concat(params, axis=1)
    return params


def get_run_data(mypath):
    onlyfiles = [
        f
        for f in listdir(mypath + "/GeneratorDataModule/")
        if isdir(join(mypath + "/GeneratorDataModule/", f))
    ]
    out = {}
    for x in onlyfiles:
        stack = [
            f
            for f in listdir(mypath + "/GeneratorDataModule/" + x + "/Architecture_PL")
            if ".p" in f
        ]
        res = {}
        for y in stack:
            res[y.split(".")[0]] = pickle.load(
                open(
                    mypath + "/GeneratorDataModule/" + x + "/Architecture_PL/" + y, "rb"
                )
            )
        out[x] = res
    return out


def load_study_results(mypath):
    onlyfiles = [f for f in listdir(mypath)]
    # group into archetyped
    groups = {}
    for ty in ["mlp", "unid", "bidirectional", "convM", "transformer"]:
        groups[ty] = [x for x in onlyfiles if ty in x]
    # make param table
    rm = []
    for x in groups:
        if len(groups[x]) == 0:
            rm.append(x)
    [groups.pop(x) for x in rm]

    scores = {}
    for x in groups:
        scores[x] = {}
        for n in groups[x]:
            curve = get_run_data(mypath + "/" + n)
            for c in curve:
                scores[x][c] = curve[c]
    return scores


def calc_roc_statistics(samples, test_set2=True):
    roc = torchmetrics.classification.BinaryROC()
    auroc = torchmetrics.classification.BinaryAUROC()
    out = {}
    for key in samples:
        out[key] = {}
        if test_set2:
            sets = ["train", "val", "test", "test2"]
        else:
            sets = ["train", "val", "test"]

        for ds in sets:
            #  not calculated.
            if len(samples[key]) == 1:
                out[key][ds] = (None, None)
            else:
                roc_out = roc(
                    preds=samples[key][ds + "_pred"],
                    target=samples[key][ds + "_lab"].type(torch.int64),
                )
                auroc_out = auroc(
                    preds=samples[key][ds + "_pred"],
                    target=samples[key][ds + "_lab"].type(torch.int64),
                )
                out[key][ds] = (roc_out, auroc_out)
    return out


def get_best_hps(results, hps):
    out = {}
    for x in results:
        best = 0
        index = "None"
        for sample in results[x]:
            if results[x][sample]["val"][1] > best:
                index = sample
                best = results[x][sample]["val"][1]
        out[x] = (index, best)
    stack = []
    for x in out:
        params = hps[x][out[x][0]]
        params["Roc"] = out[x][1]
        stack.append(params)

    return pd.concat(stack, axis=1)


def get_best_hps(results, hps, condition=False, select_on="val"):
    out = {}
    for x in results:
        best = 0
        index = "None"
        for sample in results[x]:
            if results[x][sample]["val"][1] > best:
                if condition:
                    if hps[x][sample].loc[condition]:
                        index = sample
                        best = results[x][sample]["val"][1]
                        save = results[x][sample]

                else:
                    index = sample
                    best = results[x][sample]["val"][1]
                    save = results[x][sample]

        out[x] = (
            index,
            save["train"][1].detach().numpy(),
            save["val"][1].detach().numpy(),
            save["test"][1].detach().numpy(),
            save["test2"][1].detach().numpy(),
        )
    stack = []
    for x in out:
        params = hps[x][out[x][0]]
        params["Roc_train"] = out[x][1]
        params["Roc_val"] = out[x][2]
        params["Roc_test"] = out[x][3]
        params["Roc_test2"] = out[x][4]
        params["run"] = params.name
        params.name = x
        stack.append(params)
    full = pd.concat(stack, axis=1)
    return full.loc[
        [
            "run",
            "n_vars",
            "max_lags",
            "regression_head",
            "corr_input",
            "model.corr_regularization",
            "data.batch_size",
            "model.model_type",
            "model.optimizer_lr",
            "model.weight_decay",
            "Roc_train",
            "Roc_val",
            "Roc_test",
            "Roc_test2",
        ]
    ]


def add_hp_scoring(results, hps):
    for x in results:
        for sample in results[x]:
            hps[x].loc["Roc_train", sample] = (
                results[x][sample]["train"][1].detach().numpy()
            )
            hps[x].loc["Roc_val", sample] = (
                results[x][sample]["val"][1].detach().numpy()
            )
            hps[x].loc["Roc_test", sample] = (
                results[x][sample]["test"][1].detach().numpy()
            )
            hps[x].loc["Roc_test2", sample] = (
                results[x][sample]["test2"][1].detach().numpy()
            )
    return hps


@timing
def corr_ci(ds, max_lags=3):
    # ds should be a 3 dimensional tensor: batch, ts length, vars
    return transform_corr_to_y(
        torch.abs(lagged_batch_corr(ds, max_lags)), max_lags, ds.shape[2]
    )


def roc_curve(out, lab, roc, auroc):
    roc_out = roc(preds=torch.Tensor(out), target=lab.type(torch.int64))
    auroc_out = auroc(preds=torch.Tensor(out), target=lab.type(torch.int64))
    return roc_out, auroc_out


def load_river_data():
    # load the files
    a = pd.read_csv(
        "data/data_raw/river_discharge_data/data_dillingen.csv",
        on_bad_lines="skip",
        skiprows=10,
        sep=";",
    )  # donau
    b = pd.read_csv(
        "data/data_raw/river_discharge_data/data_kempten.csv",
        on_bad_lines="skip",
        skiprows=10,
        sep=";",
    )  # iller
    c = pd.read_csv(
        "data/data_raw/river_discharge_data/data_lenggries.csv",
        on_bad_lines="skip",
        skiprows=10,
        sep=";",
    )  # isar
    # rename for merge
    a.rename(columns={x: x + "_D" for x in a if x != "Datum"}, inplace=True)
    b.rename(columns={x: x + "_K" for x in b if x != "Datum"}, inplace=True)
    c.rename(columns={x: x + "_L" for x in c if x != "Datum"}, inplace=True)
    # merge the files together
    j = a.merge(b, on="Datum").merge(c, on="Datum")
    # drop irrelevant cols and reindex with date.
    j.drop(columns=[x for x in j.columns if "Prüf" in x], inplace=True)
    j.index = j.Datum
    j.drop(columns="Datum", inplace=True)
    # reformat to float.
    for x in j.columns:
        j[x] = j[x].str.replace(",", ".")
    data = j.astype(float)
    data = torch.unsqueeze(
        torch.Tensor(data[[x for x in data.columns if "Mittelwert" in x]].values), 0
    )
    # var 2 causes var 1
    lab = torch.Tensor([[[1, 1, 0], [0, 1, 0], [0, 0, 1]]])
    return data, lab


def var_threshold_curve(out, lab):
    stack = []
    for thresh in range(20):
        tp, fp, tn, fn = binary_metrics(out, lab, link_threshold=thresh / 20)
        f1 = tp / (tp + 0.5 * (fp + fn))
        stack.append(f1)
    return torch.Tensor(stack)


@timing
def var_ci(ds, max_lags=3):
    n_vars = ds.shape[2]
    out = []
    for n, problem in enumerate(ds):
        # fit var with appropriate max lags
        res = VAR(problem.numpy()).fit(max_lags)
        # convert to bool and throw away intersection
        pred = np.abs(res.params[1:])
        # reformat to original caused causing lag:
        stack = []
        for lag_step in range(0, pred.shape[0], n_vars):
            # select lag subset of params
            l = pred[lag_step : lag_step + n_vars].T
            stack.append(l)
        # swap the axis to put lag in the back and flip the lag dimension
        y_ = np.flip(np.moveaxis(np.array(stack), 0, 2), axis=2)
        # calc standard metricsd.
        out.append(torch.Tensor(y_.copy()))
    out = torch.stack(out, axis=0)
    return out


def minmax_with_batch(data):
    b = data.shape[1]
    c = data.shape[2]
    maxi = (
        torch.amax(data, dim=[1, 2])
        .repeat(b * c)
        .reshape(b * c, -1)
        .T.reshape(-1, b, c)
    )
    mini = (
        torch.amin(data, dim=[1, 2])
        .repeat(b * c)
        .reshape(b * c, -1)
        .T.reshape(-1, b, c)
    )
    rescale = (data - mini) / (maxi - mini)
    return rescale


def add_hp_scoring(results, hps, test_set2=True):
    # Adds performance scores to the hp table and returns the table
    for x in results:
        for sample in results[x]:
            hps[x].loc["Roc_train", sample] = (
                results[x][sample]["train"][1].detach().numpy()
            )
            hps[x].loc["Roc_val", sample] = (
                results[x][sample]["val"][1].detach().numpy()
            )
            hps[x].loc["Roc_test", sample] = (
                results[x][sample]["test"][1].detach().numpy()
            )
            if test_set2:
                hps[x].loc["Roc_test2", sample] = (
                    results[x][sample]["test2"][1].detach().numpy()
                )
    return hps


def calc_mae(raw):
    out = {}
    for x in raw:
        out[x] = {}
        out[x]["train"] = [
            None,
            np.abs(raw[x]["train_lab"] - raw[x]["train_pred"]).mean(),
        ]
        out[x]["val"] = [None, np.abs(raw[x]["val_lab"] - raw[x]["val_pred"]).mean()]
        out[x]["test"] = [None, np.abs(raw[x]["test_lab"] - raw[x]["test_pred"]).mean()]
    return out


def parse_best_to_commands(hps):
    summary = hps.T.sort_values("Roc_val", ascending=False).T
    # just keep the best run.
    summary = summary[summary.columns[0]]
    # base command.
    base = "python train.py"
    # all relevant parameters:
    fixed = [
        "max_lags",
        "n_vars",
        "model.model_type",
        "data.ds_name",
        "corr_input",
        "regression_head",
        "model.corr_regularization",
        "data.batch_size",
        "model.optimizer_lr",
        "model.weight_decay",
    ]
    # add precise specifications for best run
    for x in fixed:
        base += " " + x + "=" + str(summary.loc[x])
    # hack the model size since hydra doesnt save default specifications
    if summary["model.d_model"] == 32:
        base += " model=small.yaml"
    elif summary["model.d_model"] == 128:
        base += " model=medium.yaml"
    elif summary["model.d_model"] == 256:
        base += " model=big.yaml"
    elif summary["model.d_model"] == 512:
        base += " model=deep.yaml"
    else:
        print("SIZE UNKNOWN")
        return 0
    return base


def corr_run_full_ds(ds):
    roc = torchmetrics.classification.BinaryROC()
    auroc = torchmetrics.classification.BinaryAUROC()
    stack_pred = []
    stack_lab = []
    for x, y in ds:
        stack_pred.append(corr_ci(x, max_lags=y.shape[3])[0])
        stack_lab.append(y)
    stack_pred = torch.concat(stack_pred, axis=0)
    stack_lab = torch.concat(stack_lab, axis=0)
    out = roc_curve(stack_pred, stack_lab, roc=roc, auroc=auroc)
    return out


def corr_calc_aerosol(data):
    stack_pred = []
    x, y = data
    stack_pred.append(corr_ci(x, max_lags=1)[0])
    stack_pred = torch.concat(stack_pred, axis=0)
    return (
        stack_pred[:, 0, 1].max(axis=1)[0] > stack_pred[:, 1, 0].max(axis=1)[0]
    ).sum() / len(stack_pred)


def corr_run_full_ds_masked_diagonal(d, max_lags=1, size=15):
    roc = torchmetrics.classification.BinaryROC()
    auroc = torchmetrics.classification.BinaryAUROC()
    stack_pred = []
    stack_lab = []
    for x, y in d:
        stack_pred.append(corr_ci(x, max_lags=max_lags)[0])
        stack_lab.append(y)
    stack_pred = torch.concat(stack_pred, axis=0)
    stack_lab = torch.concat(stack_lab, axis=0)

    stack_pred_cleaned = []
    stack_lab_cleaned = []
    # remove the diagonal from calculation.
    for x in stack_pred:
        stack_pred_cleaned.append(
            x[:, :, 0].flatten()[~torch.eye(size, size).flatten().bool()]
        )
    # remove the diagonal from calculation.
    for x in stack_lab:
        stack_lab_cleaned.append(
            x[:, :, 0].flatten()[~torch.eye(size, size).flatten().bool()]
        )
    return roc_curve(
        torch.concat(stack_pred_cleaned),
        torch.concat(stack_lab_cleaned),
        roc=roc,
        auroc=auroc,
    )


def var_run_full_ds(ds):
    roc = torchmetrics.classification.BinaryROC()
    auroc = torchmetrics.classification.BinaryAUROC()
    out = []
    lab = []
    for x, y in ds:
        out.append(var_ci(x, max_lags=y.shape[3])[0])
        lab.append(y)
    out = torch.concat(out, axis=0)
    lab = torch.concat(lab, axis=0)
    return roc_curve(out, lab, roc=roc, auroc=auroc)


def var_calc_aerosol(data):
    stack_pred = []
    x, y = data
    res, time = var_ci(x, max_lags=1)
    stack_pred.append(res)
    stack_pred = torch.concat(stack_pred, axis=0)
    return (stack_pred[:, 0, 1, -1] > stack_pred[:, 1, 0, -1]).sum() / len(stack_pred)


def var_run_full_ds_masked_diagonal(d, max_lags=1, size=15):
    roc = torchmetrics.classification.BinaryROC()
    auroc = torchmetrics.classification.BinaryAUROC()
    stack_pred = []
    stack_lab = []
    for x, y in d:
        stack_pred.append(var_ci(x, max_lags=max_lags)[0])
        stack_lab.append(y)
    stack_pred = torch.concat(stack_pred, axis=0)
    stack_lab = torch.concat(stack_lab, axis=0)

    stack_pred_cleaned = []
    stack_lab_cleaned = []
    # remove the diagonal from calculation.
    for x in stack_pred:
        stack_pred_cleaned.append(
            x[:, :, 0].flatten()[~torch.eye(size, size).flatten().bool()]
        )
    # remove the diagonal from calculation.
    for x in stack_lab:
        stack_lab_cleaned.append(
            x[:, :, 0].flatten()[~torch.eye(size, size).flatten().bool()]
        )
    return roc_curve(
        torch.concat(stack_pred_cleaned),
        torch.concat(stack_lab_cleaned),
        roc=roc,
        auroc=auroc,
    )
