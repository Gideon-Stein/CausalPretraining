"""Re-score every std run of the thesis experiments from its checkpoint.

    python thesis/rescore.py --mode eval              # proper inference (model.eval())
    python thesis/rescore.py --mode legacy            # reproduces the stored CSVs
    python thesis/rescore.py --mode eval --only kuramoto joint_big

Unlike the committed CSVs, this walks all run folders listed in manifest/runs.yaml, so runs
that have a checkpoint but were never scored are included. Writes
thesis/results/rescored_<mode>.csv with one row per run and thesis names.

Eval mode writes the thesis metrics per test set (common.metrics): ts<k>_nshd and ts<k>_nauprc, per graph
and averaged over the graphs with at least one edge (ts<k>_no_edges graphs excluded), and ts<k>_auroc, pooled
over the scored entries. Kuramoto is scored without self-links and Joint at the true size of each sample,
so the Joint AUROC is lower than the stored one, which includes the padding. Predictions are cached in
paths.yaml:derived_root/predictions. Legacy mode only reproduces the stored AUROC (all entries, incl. the
Kuramoto self-links).
"""
import argparse

import pandas as pd
import yaml

from common import (ARCHS, PATHS, THESIS, auroc, best_checkpoint, legacy_batch_size, load_model, predict,
                    run_config, score_split, test_set_path)

MANIFEST = yaml.safe_load(open(THESIS / "manifest" / "runs.yaml"))


def root_of(name):
    """cp_1 (runs_root) or, for entries with `root: new`, the thesis runs (new_runs_root)."""
    return PATHS["new_runs_root"] if MANIFEST["std"][name].get("root") == "new" else PATHS["runs_root"]


def runs_of(name):
    """(run_dir, overrides) of every run below the folders of an std experiment.

    Values set in an hparams_search file instead of on the command line (the cm_fixed runs) are
    taken from the composed config."""
    entry = MANIFEST["std"][name]
    for folder in entry["runs"]:
        for ov_path in sorted((root_of(name) / folder).rglob(".hydra/overrides.yaml")):
            run_dir = ov_path.parent.parent
            ov = dict(x.split("=", 1) for x in yaml.safe_load(open(ov_path)))
            cfg = run_config(run_dir)
            for k in ("model.model_type", "model.optimizer_lr", "data.batch_size", "model.weight_decay",
                      "model.corr_regularization", "corr_input", "regression_head", "data.ds_name"):
                if k not in ov:
                    ov[k] = str(cfg[k.split(".")[0]][k.split(".")[1]] if "." in k else cfg[k])
            yield run_dir, ov


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["eval", "legacy"], required=True)
    parser.add_argument("--only", nargs="*", default=None, help="std experiments from manifest/runs.yaml")
    args = parser.parse_args()

    # scaling_deep and scaling_lcm share one cp_1 folder; score each run once
    names = args.only or [n for n in MANIFEST["std"] if n != "scaling_lcm"]
    rows, seen = [], set()
    for name in names:
        entry = MANIFEST["std"][name]
        for run_dir, ov in runs_of(name):
            if run_dir in seen:
                continue
            seen.add(run_dir)
            row = {
                "experiment": name,
                "run_dir": str(run_dir.relative_to(root_of(name))),
                "arch": ARCHS[ov["model.model_type"]],
                "dataset": ov.get("data.ds_name", "base"),
                "size": ov.get("model", "small.yaml").removesuffix(".yaml"),
                "seed": int(ov["seed"]),
                **{k: ov[k] for k in ("model.optimizer_lr", "data.batch_size", "model.weight_decay",
                                      "model.corr_regularization", "corr_input", "regression_head")},
            }
            ckpt = best_checkpoint(run_dir)
            if ckpt is None:
                print(f"[skip] no checkpoint: {row['run_dir']}")
                continue
            model = None
            for ts in (1, 2):
                path = test_set_path(row["dataset"], ts)
                if not path.exists():  # Kuramoto has no in-distribution Test-Set 1
                    continue
                if args.mode == "legacy":
                    model = model or load_model(ckpt)
                    pred, lab = predict(model, path, binary=run_config(run_dir)["data"]["binary"], legacy=True,
                                        batch_size=legacy_batch_size(ts, entry.get("legacy_bs")))
                    row[f"ts{ts}_auroc"] = auroc(pred, lab)
                    continue
                m = score_split(run_dir, path, lambda: model or load_model(ckpt))
                for k in ("nshd", "nauprc", "auroc", "no_edges"):
                    row[f"ts{ts}_{k}"] = m[k]
            nan = float("nan")
            print(f"[{name}] {row['arch']:4s} {row['size']:6s} {row['dataset']:16s} seed={row['seed']} "
                  + " ".join(f"TS{ts} AUROC={row.get(f'ts{ts}_auroc', nan):.4f} nSHD={row.get(f'ts{ts}_nshd', nan):.4f} "
                             f"nAUPRC={row.get(f'ts{ts}_nauprc', nan):.4f}" for ts in (1, 2)), flush=True)
            rows.append(row)

    out = THESIS / "results" / f"rescored_{args.mode}.csv"
    out.parent.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists() and args.only:
        old = pd.read_csv(out)
        df = pd.concat([old[~old["experiment"].isin(names)], df])
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} runs to {out}")


if __name__ == "__main__":
    main()
