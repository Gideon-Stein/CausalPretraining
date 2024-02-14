import argparse
import pickle
import os
import pandas as pd
import torchmetrics

from data.generator import load_deterministic_ds
from helpers.tools import (
    corr_calc_aerosol,
    corr_ci,
    corr_run_full_ds,
    corr_run_full_ds_masked_diagonal,
    load_river_data,
    roc_curve,
)
from helpers.tools import (
    load_river_data,
    roc_curve,
    var_calc_aerosol,
    var_ci,
    var_run_full_ds,
)


# Specify which baselines should be calculated.
# PLEASE use the second environment for the PCMCI results!!


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcmci", action="store_true")
    parser.add_argument("--var", action="store_true")
    parser.add_argument("--corr", action="store_true")
    parser.add_argument("--synth", action="store_true")
    parser.add_argument("--others", action="store_true")
    parser.add_argument("--speed", action="store_true")

    args = parser.parse_args()

    synth1 = ["SL", "ML", "SNL", "MNL", "LNL", "XLNL", "joint"]
    synth2 = ["scale_up", "scale_up_2", "scale_up_3", "scale_up_4", "scale_up_5"]

    if not os.path.exists("scores"):
        os.makedirs("scores")

    if args.corr:
        if args.synth:
            save_paths = [
                "scores/corr_test_auroc.p",
                "scores/corr_test_scale_up_auroc.p",
            ]
            # Te is full out of distribution while Te2 is in distribution data.
            for n, exp in enumerate(
                [
                    synth1,
                    synth2,
                ]
            ):
                save = {}
                print("Calc Corr Synth " + str(n))
                for ds in exp:
                    print(ds)
                    tr, val, te = load_deterministic_ds(ds, corr_input=False)
                    a = corr_run_full_ds(tr)
                    b = corr_run_full_ds(val)
                    c = corr_run_full_ds(te)
                    _, _, te2 = load_deterministic_ds(
                        ds_name=ds + "_additional", corr_input=False
                    )
                    d = corr_run_full_ds(te2)
                    save[ds] = [a, b, c, d]
                pickle.dump(save, open(save_paths[n], "wb"))

        if args.others:
            print("Calc Corr Others")
            # Kuramoto
            tr, val, te = load_deterministic_ds("kuramoto", corr_input=False)
            a = corr_run_full_ds_masked_diagonal(tr, size=5)
            b = corr_run_full_ds_masked_diagonal(val, size=5)
            c = corr_run_full_ds_masked_diagonal(te, size=5)
            pickle.dump([a, b, c], open("scores/kuramoto_corr_test_auroc.p", "wb"))

            # River data
            roc = torchmetrics.classification.BinaryROC()
            auroc = torchmetrics.classification.BinaryAUROC()
            data, lab = load_river_data()
            corr = corr_ci(data, max_lags=1)[0]
            result = roc_curve(corr.max(axis=3)[0], lab, roc, auroc)
            pickle.dump(result, open("scores/river_corr_test_auroc.p", "wb"))

            # Aerosol:
            data = pickle.load(
                open("data/deterministic_ds/aerosol/simple_test.p", "rb")
            )
            res = corr_calc_aerosol(data)
            pickle.dump(res, open("scores/aerosol_corr_test_auroc.p", "wb"))

        if args.speed:
            print("Calc Corr Inference speed")
            # inference speed
            out = []
            for _ in range(100):
                stack = []
                for ds in synth2:
                    tr, val, te = load_deterministic_ds(ds, corr_input=False)
                    te = next(iter(te))[0]
                    res, secs = corr_ci(te, max_lags=3)
                    stack.append(secs)
                out.append(stack)
            table = pd.concat(
                [pd.DataFrame(out).mean(), pd.DataFrame(out).std()], axis=1
            )
            table.columns = ["M_C", "Std_C"]
            table.to_csv("scores/compute_C.csv")

    if args.var:
        if args.synth:

            save_paths = ["scores/var_test_auroc.p", "scores/var_test_scale_up_auroc.p"]

            for n, exp in enumerate(
                [
                    synth1,
                    synth2,
                ]
            ):
                save = {}
                print("Calc VAR Synth " + str(n))
                for ds in exp:
                    tr, val, te = load_deterministic_ds(ds, corr_input=False)
                    a = var_run_full_ds(tr)
                    b = var_run_full_ds(val)
                    c = var_run_full_ds(te)
                    _, _, te2 = load_deterministic_ds(
                        ds_name=ds + "_additional", corr_input=False
                    )
                    d = var_run_full_ds(te2)
                    save[ds] = [a, b, c, d]
                pickle.dump(save, open(save_paths[n], "wb"))

        if args.others:
            print("Calc Var Others")

            # Kuramoto
            for ds in ["kuramoto"]:
                tr, val, te = load_deterministic_ds(ds, corr_input=False)
                a = var_run_full_ds(tr)
                b = var_run_full_ds(val)
                c = var_run_full_ds(te)
            pickle.dump([a, b, c], open("scores/var_test_kuramoto_auroc.p", "wb"))

            # River data
            roc = torchmetrics.classification.BinaryROC()
            auroc = torchmetrics.classification.BinaryAUROC()
            data, lab = load_river_data()
            corr = var_ci(data, max_lags=1)[0]
            result = roc_curve(corr.max(axis=3)[0], lab, roc, auroc)
            pickle.dump(result, open("scores/river_var_test_auroc.p", "wb"))

            # Aerosol:
            data = pickle.load(
                open("data/deterministic_ds/aerosol/simple_test.p", "rb")
            )
            res = var_calc_aerosol(data)
            pickle.dump(res, open("scores/aerosol_var_test_auroc.p", "wb"))

        if args.speed:
            print("Calc Var Inference speed")
            # inference speed
            out = []
            for _ in range(100):
                stack = []
                for ds in synth2:
                    tr, val, te = load_deterministic_ds(ds, corr_input=False)
                    te = next(iter(te))[0]

                    res, secs = var_ci(te, max_lags=3)
                    stack.append(secs)
                out.append(stack)

            table = pd.concat(
                [pd.DataFrame(out).mean(), pd.DataFrame(out).std()], axis=1
            )
            table.columns = ["M_V", "Std_V"]
            table.to_csv("scores/compute_V.csv")

    if args.pcmci:
        print("Please use second environment for pcmci computations.")
        print("Also this takes a while...")
        from helpers.tigramite_tools import (
            pcmci_ci,
            pcmci_calc_aerosol,
            pcmci_run_full_ds,
        )
        from tigramite.independence_tests.parcorr_wls import  ParCorr

        if args.synth:
            print("Calc PCMCI Synthetic")

            save_paths = [
                "scores/pcmci_test_auroc.p",
                "scores/pcmci_test_auroc_joint.p",
                "scores/pcmci_test_scale_up_auroc.p",
            ]

            roc = torchmetrics.classification.BinaryROC()
            auroc = torchmetrics.classification.BinaryAUROC()

            for n, exp in enumerate(
                [
                    synth1[:-1],
                    synth1[-1:],
                    synth2,
                ]
            ):
                save = {}
                for ds in exp:
                    tr, val, te = load_deterministic_ds(ds, corr_input=False)
                    out, lab = pcmci_run_full_ds(te)
                    d = roc_curve(1 - out, lab, roc, auroc)
                    tr, val, te = load_deterministic_ds(
                        ds + "_additional", corr_input=False
                    )
                    out, lab = pcmci_run_full_ds(te)
                    d2 = roc_curve(
                        1 - out, lab, roc, auroc
                    )  # we reverse the p values to calculate the roc curve (only based on ordering)
                    save[ds] = [d, d2]
                pickle.dump(save, open(save_paths[n], "wb"))

        if args.others:
            print("Calc PCMCI Others")
            save = {}
            for ds in ["kuramoto"]:
                print(ds)
                tr, val, te = load_deterministic_ds(ds,corr_input=False)
                out, lab = pcmci_run_full_ds(te)
                d = roc_curve(1-out,lab, roc, auroc)
                save[ds] = [d]
            pickle.dump(save, open("scores/pcmci_test_kuramoto_auroc.p", "wb"))

            data,lab = load_river_data()
            c_test  = ParCorr()
            roc = torchmetrics.classification.BinaryROC()
            auroc = torchmetrics.classification.BinaryAUROC()
            corr = pcmci_ci(data,c_test, max_lags=3)[0]
            result = roc_curve(1- corr.min(axis=3),lab, roc,auroc)
            pickle.dump(result, open("scores/river_pcmci_test_auroc.p", "wb"))
            result

            # Aerosol:  (cut the ts at 50 to make this somehow runnable.)
            data = pickle.load(open("data/deterministic_ds/aerosol/simple_test.p", "rb"))
            res = pcmci_calc_aerosol([data[0][:,:100,:], data[1][:,:100,:]])
            print(res)
            pickle.dump(res, open("scores/aerosol_var_test_auroc.p", "wb"))

        if args.speed:
            print("Calc PCMCI Inference speed")
            # inference speed
            c_test  = ParCorr()
            out = []
            for ds in synth2: # "big_base", "big_base_mirror", "poly", "big_poly", "nl", "big_nl"
                stack = []
                tr, val, te = load_deterministic_ds(ds,corr_input=False)
                for _ in range(10):
                    sample = next(iter(te))[0][:2]
                    res,secs = pcmci_ci(sample, c_test, max_lags=3)
                    stack.append(secs)
                out.append(stack)
                table = pd.concat([pd.DataFrame(out).T.mean(), pd.DataFrame(out).T.std()],axis=1)
                table.columns = ["M_P", "Std_P"]
                table.to_csv("scores/compute_P.csv")

if __name__ == "__main__":
    main()
