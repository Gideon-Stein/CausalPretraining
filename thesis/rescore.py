"""Re-score every std run of the thesis experiments from its checkpoint.

    python thesis/rescore.py --mode eval              # proper inference (model.eval())
    python thesis/rescore.py --mode legacy            # reproduces the stored CSVs
    python thesis/rescore.py --mode eval --only kuramoto joint_big

Unlike the committed CSVs, this walks all run folders listed in manifest/runs.yaml, so runs
that have a checkpoint but were never scored are included. Writes
thesis/results/rescored_<mode>.csv with one row per run and thesis names.
"""
import argparse

import pandas as pd
import yaml

from common import (ARCHS, PATHS, THESIS, auroc, best_checkpoint, legacy_batch_size, load_model,
                    predict, run_config, test_set_path)

MANIFEST = yaml.safe_load(open(THESIS / "manifest" / "runs.yaml"))


def runs_of(name):
    """(run_dir, overrides) of every run below the cp_1 folders of an std experiment."""
    entry = MANIFEST["std"][name]
    for folder in entry["runs"]:
        for ov in sorted((PATHS["runs_root"] / folder).rglob(".hydra/overrides.yaml")):
            yield ov.parent.parent, dict(x.split("=", 1) for x in yaml.safe_load(open(ov)))


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
                "run_dir": str(run_dir.relative_to(PATHS["runs_root"])),
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
            model = load_model(ckpt)
            binary = run_config(run_dir)["data"]["binary"]
            for ts in (1, 2):
                path = test_set_path(row["dataset"], ts)
                if not path.exists():  # Kuramoto has no in-distribution Test-Set 1
                    continue
                kwargs = {"legacy": True, "batch_size": legacy_batch_size(ts, entry.get("legacy_bs"))} \
                    if args.mode == "legacy" else {}
                row[f"ts{ts}_auroc"] = auroc(*predict(model, path, binary=binary, **kwargs))
            print(f"[{name}] {row['arch']:4s} {row['size']:6s} {row['dataset']:16s} seed={row['seed']} "
                  f"TS1={row.get('ts1_auroc', float('nan')):.4f} TS2={row.get('ts2_auroc', float('nan')):.4f}",
                  flush=True)
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
