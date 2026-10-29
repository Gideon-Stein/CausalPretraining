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
recomputed AUROC.
"""
import argparse
import pickle

import pandas as pd
from torch.utils.data import DataLoader

from common import PATHS, SCALE_UP, THESIS, deterministic_ds
from helpers.tools import corr_run_full_ds, corr_run_full_ds_masked_diagonal

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


def recompute(dataset, split):
    if dataset == "kuramoto":
        file_split = {"train": "train", "val": "val", "ts2": "test"}[split]
        return float(corr_run_full_ds_masked_diagonal(loader("kuramoto", file_split), size=5)[1])
    folder = dataset + "_additional" if split == "ts1" else dataset
    file_split = "test" if split in ("ts1", "ts2") else split
    return float(corr_run_full_ds(loader(folder, file_split))[1])


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
        print(f"{r['dataset']:16s} {r['split']:5s} stored {r['stored']:.4f} "
              f"recomputed {df.loc[i, 'recomputed']:.4f}", flush=True)

    df["abs_diff"] = (df["stored"] - df["recomputed"]).abs()
    out = THESIS / "results" / "ct_baseline.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nmax |stored - recomputed| = {df['abs_diff'].max():.2e} over {df['recomputed'].notna().sum()} "
          f"splits; wrote {out}")


if __name__ == "__main__":
    main()
