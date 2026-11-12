"""Correlation-thresholding (CT) reference: recompute on the original data and compare
with the scores stored for the paper.

    python thesis/ct_baseline.py            # val + test splits (fast)
    python thesis/ct_baseline.py --train    # also the training splits

Uses the same functions as calc_baselines.py --corr (corr_run_full_ds, and the masked
diagonal variant for Kuramoto), but reads the data from paths.yaml:data_root instead of
the relative data/deterministic_ds. The stored pickles (paths.yaml:baseline_scores) hold,
per dataset, [train, val, <ds>/simple_test.p, <ds>_additional/simple_test.p] as
(roc curve, auroc); Kuramoto has [train, val, test].

Writes thesis/results/ct_baseline.csv: dataset, split (train/val/ts1/ts2), stored and
recomputed AUROC (pooled over all entries, as in the paper), and the thesis metrics nshd, nauprc (per graph,
common.metrics: Kuramoto without self-links, Joint at the true size of each sample, graphs without edges
excluded) with auroc_scored (pooled over the scored entries only).


NOTABLY, WE IGNORE THE OTHER BASELINES HERE AS WE DO THEM LATER ON ON THE REAL BENCHMARKS
GVAR IN THEORIGINAL PAPER IS WORSE BECAUSE OF THE COEFFICIENT ESTIMATION THAT IS OFTEN HIGH AS THE VARIANCE OF THE FILLING IS REALLY LOW, 
GOOD TO KEEP THAT IN MIND:

"""
import argparse
import pickle

import pandas as pd
from torch.utils.data import DataLoader

import torch

from common import PATHS, SCALE_UP, THESIS, deterministic_ds, entry_mask, metrics
from helpers.tools import corr_ci, corr_run_full_ds, corr_run_full_ds_masked_diagonal

SINGLE_JOINT = ["base", "big_base", "big_base_mirror", "poly", "big_poly", "nl", "big_nl", "joint"]
STORED_SPLITS = ["train", "val", "ts2", "ts1"]


def stored():
    rows = []
    for name, experiment in [("corr_test_auroc.p", "single_joint"),
                             ("corr_test_scale_up_auroc.p", "scaling"),
                             ("kuramoto_corr_test_auroc.p", "kuramoto")]:
        raw = pickle.load(open(PATHS["baseline_scores"] / name, "rb"))
        if not isinstance(raw, dict):
            raw = {"kuramoto": raw}
        for ds, results in raw.items():
            for split, res in zip(STORED_SPLITS, results):
                rows.append({"experiment": experiment, "dataset": ds, "split": split,
                             "stored": float(res[1])})
    return pd.DataFrame(rows)


def loader(folder, split):
    """Same settings as load_deterministic_ds(ds, corr_input=False) in calc_baselines.py."""
    ds = deterministic_ds(ds_path=str(PATHS["data_root"] / folder / f"simple_{split}.p"),
                          binary=True, corr_input=False)
    return DataLoader(ds, batch_size=500, shuffle=False)


def split_file(dataset, split):
    if dataset == "kuramoto":
        return "kuramoto", {"train": "train", "val": "val", "ts2": "test"}[split]
    return (dataset + "_additional" if split == "ts1" else dataset), ("test" if split in ("ts1", "ts2") else split)


def recompute(dataset, split):
    """Pooled AUROC with the original functions (comparable to the stored value)."""
    folder, file_split = split_file(dataset, split)
    if dataset == "kuramoto":
        return float(corr_run_full_ds_masked_diagonal(loader(folder, file_split), size=5)[1])
    return float(corr_run_full_ds(loader(folder, file_split))[1])


def thesis_metrics(dataset, split):
    """nSHD, nAUPRC per graph (and AUROC over the scored entries) of the same CT scores (|lagged corr.|)."""
    folder, file_split = split_file(dataset, split)
    preds, labs = [], []
    for x, y in loader(folder, file_split):
        preds.append(corr_ci(x, max_lags=y.shape[3])[0])
        labs.append(y)
    pred, lab = torch.cat(preds), torch.cat(labs)
    m = metrics(pred, lab, entry_mask(PATHS["data_root"] / folder / f"simple_{file_split}.p", lab.shape))
    return {"nshd": m["nshd"], "nauprc": m["nauprc"], "auroc_scored": m["auroc"], "no_edges": m["no_edges"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true", help="also recompute the training splits")
    args = parser.parse_args()

    df = stored()
    assert set(df.loc[df["experiment"] == "single_joint", "dataset"]) == set(SINGLE_JOINT)
    assert set(df.loc[df["experiment"] == "scaling", "dataset"]) == set(SCALE_UP.values())
    todo = df["split"] != "train" if not args.train else df["split"].notna()
    for i in df.index[todo]:
        r = df.loc[i]
        df.loc[i, "recomputed"] = recompute(r["dataset"], r["split"])
        for k, v in thesis_metrics(r["dataset"], r["split"]).items():
            df.loc[i, k] = v
        print(f"{r['dataset']:16s} {r['split']:5s} stored {r['stored']:.4f} recomputed {df.loc[i, 'recomputed']:.4f} "
              f"nSHD {df.loc[i, 'nshd']:.4f} nAUPRC {df.loc[i, 'nauprc']:.4f}", flush=True)

    df["abs_diff"] = (df["stored"] - df["recomputed"]).abs()
    out = THESIS / "results" / "ct_baseline.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nmax |stored - recomputed| = {df['abs_diff'].max():.2e} over {df['recomputed'].notna().sum()} "
          f"splits; wrote {out}")


if __name__ == "__main__":
    main()
