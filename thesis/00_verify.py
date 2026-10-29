"""Verify that everything the thesis numbers depend on is present and unchanged.

    python thesis/00_verify.py                 # all checks, rescoring 2 runs per experiment
    python thesis/00_verify.py --data --runs   # only the cheap checks
    python thesis/00_verify.py --rescore 10    # rescore more runs

--data     checksums of the read-only data store against manifest/*.md5
--runs     every run in a committed std CSV has exactly one run folder in cp_1;
           committed grid CSVs are byte-identical to the summaries in cp_1
--arch     every std checkpoint contains the architecture its config claims
--rescore  re-score a random sample of std runs from their checkpoints on the
           original test sets and compare with the committed CSV. The stored scores
           were computed in train mode (see common.predict), so this uses legacy=True,
           which is stochastic: the stored value must lie within the range of
           LEGACY_DRAWS draws (+-TOL). Eval-mode scores are printed alongside.
"""
import argparse
import hashlib
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

from common import (PATHS, REPO, THESIS, auroc, best_checkpoint, legacy_batch_size, load_model,
                    load_results, predict, run_config, test_set_path)

RESULTS = REPO / "experimental_results"
MANIFEST = yaml.safe_load(open(THESIS / "manifest" / "runs.yaml"))
TOL = 2e-3
LEGACY_DRAWS = 8  # legacy scoring is stochastic: stored value must lie in the range of these draws (+-TOL)

# Columns that identify a std run (the overrides of the hydra sweep).
RUN_KEY = ["seed", "model.model_type", "data.ds_name", "model", "model.optimizer_lr",
           "data.batch_size", "model.corr_regularization", "corr_input", "regression_head"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_key(d):
    d = dict(d)
    d["model"] = d.get("model") if isinstance(d.get("model"), str) else "small.yaml"
    d["model.optimizer_lr"] = float(d["model.optimizer_lr"])
    return tuple(str(d.get(k, "NA")) for k in RUN_KEY)


def index_runs(folders):
    """run key -> run folders, over all hydra runs below the given cp_1 folders."""
    idx = {}
    for folder in folders:
        for ov in (PATHS["runs_root"] / folder).rglob(".hydra/overrides.yaml"):
            d = dict(x.split("=", 1) for x in yaml.safe_load(open(ov)))
            idx.setdefault(run_key(d), []).append(ov.parent.parent)
    return idx


def matched_runs(name):
    """Committed std CSV of an experiment, with the run folder of every row."""
    entry = MANIFEST["std"][name]
    df = load_results(RESULTS / entry["csv"])
    idx = index_runs(entry["runs"])
    df["run_dir"] = [idx.get(run_key(r), []) for _, r in df.iterrows()]
    return df


def check_data():
    ok = True
    for manifest in sorted((THESIS / "manifest").glob("*.md5")):
        bad = []
        for line in open(manifest):
            h, rel = line.rstrip("\n").split("  ", 1)
            p = PATHS["store_root"] / rel
            if not p.exists() or md5(p) != h:
                bad.append(rel)
        print(f"[data] {manifest.name}: {'OK' if not bad else f'{len(bad)} missing/changed'}")
        for rel in bad[:10]:
            print("       ", rel)
        ok &= not bad
    return ok


def check_runs():
    ok = True
    for name in MANIFEST["std"]:
        df = matched_runs(name)
        n = df["run_dir"].map(len)
        good = (n == 1).all()
        print(f"[runs] std/{name}: {len(df)} runs, unmatched={(n == 0).sum()}, ambiguous={(n > 1).sum()}"
              f" {'OK' if good else 'FAIL'}")
        ok &= good
    for name, entry in MANIFEST["grid"].items():
        committed = sorted((RESULTS / entry["csv"]).glob("*_summary.csv"))
        sources = {md5(p) for f in entry["runs"] for p in (PATHS["runs_root"] / f).glob("*_summary.csv")}
        missing = [p.name for p in committed if md5(p) not in sources]
        print(f"[runs] grid/{name}: {len(committed)} summaries, {len(missing)} without identical cp_1 copy"
              f" {'OK' if not missing else 'FAIL'}")
        ok &= not missing
    return ok


def built_arch(state_dict):
    keys = list(state_dict)
    head = keys[0].split(".")[1]
    if head == "rnn":
        return "bidirectional" if any("reverse" in k for k in keys) else "unidirectional"
    return {"fc1": "mlp", "convM": "convM", "enc_embedding": "transformer"}.get(head, head)


def check_arch():
    ok = True
    folders = sorted({f for e in MANIFEST["std"].values() for f in e["runs"]})
    for folder in folders:
        n, wrong, no_ckpt = 0, 0, 0
        for cfg in (PATHS["runs_root"] / folder).rglob(".hydra/config.yaml"):
            ckpt = best_checkpoint(cfg.parent.parent)
            if ckpt is None:
                no_ckpt += 1
                continue
            claimed = yaml.safe_load(open(cfg))["model"]["model_type"]
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
            n += 1
            wrong += built_arch(sd) != claimed
        print(f"[arch] {folder}: {n} checkpoints, mismatches={wrong}, runs without checkpoint={no_ckpt}"
              f" {'OK' if not wrong else 'FAIL'}")
        ok &= not wrong
    return ok


def check_rescore(k, seed=0):
    ok = True
    rng = random.Random(seed)
    for name, entry in MANIFEST["std"].items():
        df = matched_runs(name)
        df = df[df["run_dir"].map(len) == 1]
        for i in rng.sample(range(len(df)), min(k, len(df))):
            row = df.iloc[i]
            run_dir = row["run_dir"][0]
            model = load_model(best_checkpoint(run_dir))
            binary = run_config(run_dir)["data"]["binary"]
            for ts in (1, 2):
                col = f"ts{ts}_auroc"
                if col not in row or pd.isna(row[col]):
                    continue
                path = test_set_path(row["data.ds_name"], ts)
                draws = [auroc(*predict(model, path, binary=binary, legacy=True,
                                        batch_size=legacy_batch_size(ts), seed=s))
                         for s in range(LEGACY_DRAWS)]
                score_eval = auroc(*predict(model, path, binary=binary))
                good = min(draws) - TOL <= row[col] <= max(draws) + TOL
                print(f"[rescore] {name} {row['arch']:4s} seed={row['seed']} {row['data.ds_name']} TS{ts}: "
                      f"stored {row[col]:.4f} legacy [{min(draws):.4f}, {max(draws):.4f}] {'OK' if good else 'FAIL'}"
                      f" | eval {score_eval:.4f}  ({run_dir.relative_to(PATHS['runs_root'])})")
                ok &= good
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--runs", action="store_true")
    parser.add_argument("--arch", action="store_true")
    parser.add_argument("--rescore", type=int, nargs="?", const=2, default=None, metavar="N")
    args = parser.parse_args()
    run_all = not (args.data or args.runs or args.arch or args.rescore)

    results = {}
    if run_all or args.data:
        results["data"] = check_data()
    if run_all or args.runs:
        results["runs"] = check_runs()
    if run_all or args.arch:
        results["arch"] = check_arch()
    if run_all or args.rescore:
        results["rescore"] = check_rescore(args.rescore or 2)

    print("\nSUMMARY:", ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in results.items()))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
