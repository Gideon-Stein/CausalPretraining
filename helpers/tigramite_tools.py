import numpy as np
from tigramite.pcmci import PCMCI
from tigramite import data_processing as pp

import torch
import torchmetrics

from functools import wraps
from time import time
from tigramite.independence_tests.parcorr_wls import ParCorr
from tigramite.independence_tests.gpdc import GPDC


def roc_curve(out, lab, roc, auroc):
    roc_out = roc(preds=torch.Tensor(out), target=lab.type(torch.int64))
    auroc_out = auroc(preds=torch.Tensor(out), target=lab.type(torch.int64))
    return roc_out, auroc_out


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("took: %2.4f sec" % (te - ts))
        return result, te - ts

    return wrap


def tensor_to_pcmci_res(sample, c_test, max_tau):
    dataframe = pp.DataFrame(
        sample.detach().numpy().astype(float),
        datatime=np.arange(len(sample)),
        var_names=np.arange(sample.shape[1]),
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=c_test, verbosity=1)
    pcmci.verbosity = 0
    results = pcmci.run_pcmci(tau_min=0, tau_max=max_tau, pc_alpha=None)
    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results["p_matrix"], fdr_method="fdr_bh"
    )
    return q_matrix


@timing
def pcmci_ci(ds, c_test, max_lags=3):
    out = []
    for sample in ds:
        q_matrix = tensor_to_pcmci_res(sample, c_test, max_tau=max_lags)
        out.append(np.swapaxes(np.flip(q_matrix[:, :, 1:], 2), 0, 1))
    out = np.stack(out, axis=0)
    return out


def pcmci_run_full_ds(dl, nonlinear=False):
    if nonlinear:
        c_test = GPDC()
    else:
        c_test = ParCorr()
    out = []
    lab = []
    for x, y in dl:
        out.append(pcmci_ci(x, c_test, max_lags=y.shape[3])[0])
        lab.append(y)
    out = torch.Tensor(np.concatenate(out, axis=0))
    lab = torch.concat(lab, axis=0)
    return out, lab


def pcmci_calc_aerosol(data):
    c_test =  ParCorr()
    stack_pred = []
    x, y = data
    res, t = pcmci_ci(x, c_test, max_lags=1)
    stack_pred.append(torch.Tensor(res))
    stack_pred = torch.concat(stack_pred, axis=0)
    return (
        stack_pred[:, 0, 1].max(axis=1)[0] > stack_pred[:, 1, 0].max(axis=1)[0]
    ).sum() / len(stack_pred)


def pcmci_run_full_ds_masked_diagonal(d, max_lags=1, size=15):
    roc = torchmetrics.classification.BinaryROC()
    auroc = torchmetrics.classification.BinaryAUROC()
    c_test = ParCorr()
    stack_pred = []
    stack_lab = []
    for x, y in d:
        stack_pred.append(torch.Tensor(pcmci_ci(x, c_test, max_lags=max_lags)[0]))
        stack_lab.append(torch.Tensor(y))
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
