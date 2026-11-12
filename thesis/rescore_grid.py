"""Re-score the Single grid search (convergence helpers, Ch. 7.1) in eval mode.

    python thesis/rescore_grid.py                    # all runs (resumes from the cache)
    python thesis/rescore_grid.py --check 30         # reproduce stored legacy Val_Auroc on 30 runs
    python thesis/rescore_grid.py --cm-fixed         # CM grid with the corrected head (cm_fixed/single_grid)

The grid (manifest/runs.yaml: grid.single_run_1, grid.single_run_2) covers every on/off
combination of the three convergence helpers for all architectures and the six Single datasets.
The original grid scoring (slurm/slurm_calc_grid_scoring.py) scored only the validation split,
without model.eval() and with batch size 16. Here every run is scored with its best checkpoint in
eval mode on the validation split and on both test sets.

Both passes (run_1, run_2) use seed 42, so they are repeated measurements, not independent seeds.

Writes (and resumes) thesis/results/grid_eval.csv, one row per run, with <split>_nshd, <split>_nauprc (thesis
metrics, per graph, common.metrics) and <split>_auroc (pooled) for split in val, ts1, ts2. Predictions are
cached in paths.yaml:derived_root/predictions.

--cm-fixed scores the CM-only grid with the corrected head (hparams_search/thesis_cm_single_grid.yaml,
runs in new_runs_root/cm_fixed/single_grid/<dataset>/<sweep>/<n>) into results/grid_eval_cm_fixed.csv,
with the same columns plus val_mse (from the run's final_scoring.csv, the selection criterion).
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from common import (ARCHS, PATHS, REPO, THESIS, auroc, best_checkpoint, load_model, predict, run_config,
                    score_split, test_set_path)

MANIFEST = yaml.safe_load(open(THESIS / "manifest" / "runs.yaml"))
PASSES = {"run_1": "single_run_1", "run_2": "single_run_2"}
HPARAMS = ["model.optimizer_lr", "data.batch_size", "model.weight_decay", "model.model_type",
           "model.corr_regularization", "corr_input", "regression_head"]
OUT = THESIS / "results" / "grid_eval.csv"
CM_FIXED_ROOT = PATHS["new_runs_root"] / "cm_fixed" / "single_grid"
CM_FIXED_OUT = THESIS / "results" / "grid_eval_cm_fixed.csv"


def grid_runs():
    """(pass, dataset, run number, run_dir, overrides) of every grid run."""
    for pass_name, key in PASSES.items():
        for folder in MANIFEST["grid"][key]["runs"]:
            for ov_path in sorted((PATHS["runs_root"] / folder).glob("*/*/.hydra/overrides.yaml")):
                run_dir = ov_path.parent.parent
                ov = dict(x.split("=", 1) for x in yaml.safe_load(open(ov_path)))
                yield pass_name, run_dir.parent.name, run_dir.name, run_dir, ov


def cm_fixed_runs():
    """Same as grid_runs for the CM grid with the corrected head; hyperparameters from the composed config."""
    for cfg_path in sorted(CM_FIXED_ROOT.glob("*/*/*/.hydra/config.yaml")):
        run_dir = cfg_path.parent.parent
        cfg = run_config(run_dir)
        get = lambda k: cfg[k.split(".")[0]][k.split(".")[1]] if "." in k else cfg[k]
        ov = {k: str(get(k)) for k in HPARAMS}
        ov["model"] = "small" if cfg["n_vars"] == 3 else "medium"
        assert cfg["model"]["convM_head_activation"], run_dir
        yield "cm_fixed", run_dir.parent.parent.name, run_dir.name, run_dir, ov


def final_val_mse(run_dir):
    sc = pd.read_csv(Path(run_dir) / "run" / "final_scoring.csv", header=None, index_col=0)[1]
    return float(sc["val_MSE"])


def split_paths(dataset):
    return {"val": PATHS["data_root"] / dataset / "simple_val.p",
            "ts1": test_set_path(dataset, 1), "ts2": test_set_path(dataset, 2)}


def score(run_dir, dataset, legacy=False):
    if legacy:
        model = load_model(best_checkpoint(run_dir))
        path = split_paths(dataset)["val"]
        binary = run_config(run_dir)["data"]["binary"]
        return {"val_auroc": auroc(*predict(model, path, binary=binary, legacy=True, batch_size=16))}
    model = []  # loaded at most once, only on a prediction-cache miss

    def get():
        if not model:
            model.append(load_model(best_checkpoint(run_dir)))
        return model[0]
    row = {}
    for k, p in split_paths(dataset).items():
        m = score_split(run_dir, p, get)
        row.update({f"{k}_{x}": m[x] for x in ("nshd", "nauprc", "auroc")})
    return row


def stored_scores():
    """Stored (legacy) Val_Auroc per (pass, dataset, run number)."""
    base = REPO / "experimental_results"
    out = {}
    for pass_name, key in PASSES.items():
        for f in sorted((base / MANIFEST["grid"][key]["csv"]).glob("*_summary.csv")):
            d = pd.read_csv(f, index_col=0)
            for run, v in d.loc["Val_Auroc"].items():
                out[(pass_name, f.name.removesuffix("_summary.csv"), str(run))] = float(v)
    return out


def check(n):
    """Reproduce the stored legacy scores on n runs, to verify the run-to-CSV mapping."""
    stored = stored_scores()
    done = 0
    for pass_name, dataset, num, run_dir, _ in grid_runs():
        key = (pass_name, dataset, num)
        if key not in stored or best_checkpoint(run_dir) is None:
            continue
        legacy = score(run_dir, dataset, legacy=True)["val_auroc"]
        print(f"{pass_name} {dataset:16s} {num:>4s} stored={stored[key]:.4f} legacy={legacy:.4f} "
              f"diff={legacy - stored[key]:+.4f}", flush=True)
        done += 1
        if done == n:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", type=int, default=0, help="only reproduce stored legacy scores on N runs")
    parser.add_argument("--cm-fixed", action="store_true", help="score the CM grid with the corrected head")
    args = parser.parse_args()
    if args.check:
        return check(args.check)

    out, runs, root = (CM_FIXED_OUT, cm_fixed_runs(), PATHS["new_runs_root"]) if args.cm_fixed \
        else (OUT, grid_runs(), PATHS["runs_root"])
    done = set(pd.read_csv(out)["run_dir"]) if out.exists() else set()
    stored = {} if args.cm_fixed else stored_scores()
    for pass_name, dataset, num, run_dir, ov in runs:
        key = str(run_dir.relative_to(root))
        if key in done:
            continue
        if best_checkpoint(run_dir) is None:
            print(f"[skip] no checkpoint: {key}")
            continue
        row = {"pass": pass_name, "dataset": dataset, "run": num, "run_dir": key,
               "arch": ARCHS[ov["model.model_type"]], "size": ov.get("model", "small.yaml").removesuffix(".yaml"),
               **{k: ov[k] for k in HPARAMS if k != "model.model_type"},
               **score(run_dir, dataset), "val_auroc_stored": stored.get((pass_name, dataset, num))}
        if args.cm_fixed:
            row["val_mse"] = final_val_mse(run_dir)
        print(f"[{pass_name}] {dataset:16s} {num:>4s} {row['arch']:4s} val AUROC={row['val_auroc']:.4f} "
              f"nSHD={row['val_nshd']:.4f} nAUPRC={row['val_nauprc']:.4f}", flush=True)
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
    df = pd.read_csv(out)
    print(f"{len(df)} runs in {out}")


if __name__ == "__main__":
    main()
