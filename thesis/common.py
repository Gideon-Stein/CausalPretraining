"""Shared helpers for reproducing the CP v1 numbers of the thesis.

Everything that translates between the names used in the code/result files and the
names used in the thesis lives here, so that no script has to know about it.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
THESIS = REPO / "thesis"
sys.path.insert(0, str(REPO))

from data.generator import deterministic_ds  # noqa: E402
from model.model_wrapper import Architecture_PL  # noqa: E402

PATHS = {k: Path(v) for k, v in yaml.safe_load(open(THESIS / "paths.yaml")).items()}


def _load_scoring_tools():
    """TCD-Arena cd_zoo/tools/scoring_tools.py, the single metric implementation of the thesis."""
    spec = importlib.util.spec_from_file_location("tcd_scoring_tools", PATHS["tcd_scoring_tools"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scoring_tools = _load_scoring_tools()


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

# thesis dataset name -> original dataset folder
DATASETS = {
    "SL": "base",
    "ML": "big_base_mirror",
    "SNL": "poly",
    "MNL": "big_poly",
    "LNL": "nl",
    "XLNL": "big_nl",
    "Joint": "joint",
    "Kuramoto": "kuramoto",
}
SINGLE = ["SL", "ML", "SNL", "MNL", "LNL", "XLNL"]

# scaling experiment: number of variables -> original dataset folder
SCALE_UP = {5: "big_base_mirror", 7: "scale_up_2", 10: "scale_up", 12: "scale_up_4", 15: "scale_up_3"}

# preliminary-study datasets (sanity checks)
SANITY = {
    "graphs": "classify_graphs",
    "distinguish": "distinguish",
    "coefficients": "uncover_params",
    "coefficients_small": "uncover_params_small",
}

# model_type in configs -> thesis abbreviation
ARCHS = {
    "mlp": "MLP",
    "unidirectional": "uGRU",
    "bidirectional": "bGRU",
    "convM": "CM",
    "transformer": "Trf",
}

# The stored result columns are swapped w.r.t. the thesis:
# "Test1" was scored on <ds>/simple_test.p (shifted coefficients/noise) = thesis Test-Set 2,
# "Test2" on <ds>_additional/simple_test.p (in-distribution)            = thesis Test-Set 1.
RESULT_COLUMNS = {
    "Test2_AUROC": "ts1_auroc",
    "Test2_MSE": "ts1_mse",
    "Test1_Auroc": "ts2_auroc",
    "Test1_MSE": "ts2_mse",
    "Val_Auroc": "val_auroc",
    "Val_MSE": "val_mse",
    "Train_Auroc": "train_auroc",
    "Train_MSE": "train_mse",
}


def ds_folder(name):
    """Accept a thesis name (SL, Joint, ...) or an original folder name."""
    return DATASETS.get(name, name)


def test_set_path(name, test_set):
    """Path of thesis Test-Set 1 (in-distribution) or 2 (shifted) of a dataset."""
    folder = ds_folder(name)
    if test_set == 1:
        folder += "_additional"
    elif test_set != 2:
        raise ValueError(test_set)
    return PATHS["data_root"] / folder / "simple_test.p"


def load_results(csv):
    """Load a result summary CSV (one column per run) as a tidy frame with thesis names."""
    df = pd.read_csv(csv, index_col=0).T.rename(columns=RESULT_COLUMNS)
    df["arch"] = df["model.model_type"].map(ARCHS)
    for c in RESULT_COLUMNS.values():
        if c in df:
            df[c] = pd.to_numeric(df[c])
    if "seed" in df:
        df["seed"] = pd.to_numeric(df["seed"]).astype(int)
    return df


# ---------------------------------------------------------------------------
# Runs and scoring
# ---------------------------------------------------------------------------

def run_config(run_dir):
    """The run's composed hydra config with interpolations resolved, as plain dicts."""
    cfg = OmegaConf.load(Path(run_dir) / ".hydra" / "config.yaml")
    return OmegaConf.to_container(cfg, resolve=True)


def best_checkpoint(run_dir):
    """The best (not the last) checkpoint of a run, as selected by the original scoring."""
    ckpts = [p for p in Path(run_dir).rglob("*.ckpt") if "step" in p.name]
    return ckpts[0] if ckpts else None


def load_model(ckpt):
    return Architecture_PL.load_from_checkpoint(ckpt, map_location="cpu").eval()


def legacy_batch_size(test_set, overrides=None):
    """Batch size the original scoring used: 500 for Test-Set 1 (via load_deterministic_ds),
    50 for Test-Set 2 (via the run's data module with batch_size overridden to 50).
    Some CSVs were scored with other sizes; manifest/runs.yaml records them as legacy_bs."""
    return (overrides or {}).get(test_set, 500 if test_set == 1 else 50)


@torch.no_grad()
def predict(model, ds_path, binary=True, batch_size=50, legacy=False, seed=0):
    """Sigmoid edge probabilities and labels for a whole test file.

    Mirrors predict_batch in slurm/slurm_calc_final_scoring.py (classification output).

    legacy=True reproduces how the stored result CSVs were scored: the original script
    never called model.eval(), so BatchNorm used batch statistics and dropout was active,
    on shuffled batches. Scores then depend on batch size and order (use
    legacy_batch_size). legacy=False is proper inference in eval mode.
    """
    model.train(legacy)
    torch.manual_seed(seed)
    ds = deterministic_ds(
        ds_path=str(ds_path),
        binary=binary,
        regression=model.regression_head,
        corr_input=model.hparams.corr_input,
    )
    outs, labs = [], []
    for X, Y in DataLoader(ds, batch_size=batch_size, shuffle=legacy):
        out = model.model(X)
        if model.regression_head:
            out, Y = out[0], Y[0]
        outs.append(torch.sigmoid(out))
        labs.append(Y)
    return torch.cat(outs), torch.cat(labs)


def auroc(pred, lab):
    """AUROC pooled over all entries, via TCD-Arena scoring_tools.auroc (= its "AUROC Joint").

    Replaces torchmetrics BinaryAUROC, which the original scripts used; both agree to 1.5e-7 on
    95 CP and CT predictions (2026-11-11). Computed in float64.
    """
    return float(scoring_tools.auroc(lab.cpu().numpy().astype(bool), pred.detach().cpu().double().numpy()))


def off_diagonal(pred, lab):
    """Drop the self-links of tensors shaped (samples, variables, variables, lags).

    Used for Kuramoto: the label of every self-link is 0, so keeping them adds trivial negatives
    that inflate the AUROC. The CT reference (corr_run_full_ds_masked_diagonal) drops them too.
    """
    keep = ~torch.eye(lab.shape[1], dtype=torch.bool)
    return pred[:, keep], lab[:, keep]


# ---------------------------------------------------------------------------
# Thesis metrics (ch. 2.7): nSHD and nAUPRC per graph, averaged over graphs
# ---------------------------------------------------------------------------

JOINT_PAD_STD = 0.05  # padding channels are N(0, 0.01) noise; real (min-max scaled) channels vary far more


def dataset_of(ds_path):
    """Original dataset folder of a split file, without the _additional suffix."""
    return Path(ds_path).parent.name.removesuffix("_additional")


_MASKS = {}


def entry_mask(ds_path, shape):
    """Boolean mask (samples, D, D, L) of the label entries that are scored.

    Kuramoto: all but the self-links (always 0, see off_diagonal).
    Joint: three-variable samples were padded to 5 x 5 x 3 (data/generate_synthetic_ds.py --joint: two
    N(0, 0.01) channels, label padded by two effects, two causes and the largest lag in front), so only
    their true 3 x 3 x 2 block [:3, :3, 1:] is scored. They are found by the noise channels (1500 of the
    3000 samples per split, blocks 4-6, i.e. SNL, LNL, SL; checked against the single datasets).
    Everything else: all entries.
    """
    key = (str(ds_path), tuple(shape))
    if key in _MASKS:
        return _MASKS[key]
    n, d, _, lags = shape
    mask = torch.ones(shape, dtype=torch.bool)
    dataset = dataset_of(ds_path)
    if dataset == "kuramoto":
        mask &= ~torch.eye(d, dtype=torch.bool)[None, :, :, None]
    elif dataset == "joint":
        import pickle
        raw = pickle.load(open(ds_path, "rb"))
        small = torch.tensor([bool(x[0][:, 3:].std(0).max() < JOINT_PAD_STD) for x in raw])
        assert len(raw) == n and small.sum() == n // 2, (ds_path, small.sum())
        block = torch.zeros((d, d, lags), dtype=torch.bool)
        block[:3, :3, 1:] = True
        mask[small] = block
    _MASKS[key] = mask
    return mask


def graph_metrics(pred, lab, mask=None):
    """Per-graph metrics via TCD-Arena scoring_tools: one row per graph with |E|, the number of scored
    entries, nSHD (min_shd_exact) and AUPRC, nAUPRC (auprc_normalized). Both are NaN for a graph
    without edges (undefined), nAUPRC also for a complete graph."""
    pred = pred.detach().cpu().double().numpy()
    lab = lab.cpu().numpy().astype(bool)
    mask = np.ones_like(lab) if mask is None else mask.cpu().numpy()
    rows = []
    for p, l, m in zip(pred, lab, mask):
        p, l = p[m], l[m]
        _, nshd, _, _ = scoring_tools.min_shd_exact(l, p)
        ap, nap, _ = scoring_tools.auprc_normalized(l, p)
        rows.append((int(l.sum()), l.size, nshd, ap, nap))
    return pd.DataFrame(rows, columns=["n_edges", "n_entries", "nshd", "auprc", "nauprc"])


def metrics(pred, lab, mask=None):
    """Thesis metrics of a test set: nSHD and nAUPRC averaged over the graphs with at least one edge
    (graphs without edges are excluded, their count is returned), plus the AUROC pooled over all
    scored entries of all graphs (as before, for reference only)."""
    g = graph_metrics(pred, lab, mask)
    sel = (lambda t: t[mask]) if mask is not None else (lambda t: t.flatten())
    return {"nshd": g["nshd"].mean(), "nauprc": g["nauprc"].mean(), "auroc": auroc(sel(pred), sel(lab)),
            "graphs": len(g), "no_edges": int((g["n_edges"] == 0).sum())}


# Predictions are cached per run and split, so metrics can be recomputed without the models.
PRED_ROOT = PATHS["derived_root"] / "predictions"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # inference only; GPU and CPU agree to about 1e-6
RUNS_PARENT = PATHS["runs_root"].parent  # holds cp_1 and cp_1_thesis
_INPUTS = {}


def _inputs(ds_path, binary, corr_input, batch_size=50):
    """Unshuffled input batches and labels of a split, built once per setting (the correlation input is slow)."""
    key = (str(ds_path), binary, corr_input)
    if key not in _INPUTS:
        if len(_INPUTS) >= 12:  # drop the oldest setting
            _INPUTS.pop(next(iter(_INPUTS)))
        ds = deterministic_ds(ds_path=str(ds_path), binary=binary, regression=False, corr_input=corr_input)
        xs, ys = zip(*DataLoader(ds, batch_size=batch_size, shuffle=False))
        _INPUTS[key] = (list(xs), torch.cat(ys))
    return _INPUTS[key]


def cache_path(run_dir, ds_path):
    rel = Path(run_dir).resolve().relative_to(RUNS_PARENT)
    return PRED_ROOT / rel / f"{Path(ds_path).parent.name}__{Path(ds_path).stem}.pt"


@torch.no_grad()
def cached_predict(run_dir, ds_path, model=None):
    """Eval-mode edge probabilities and labels of the run's best checkpoint on a split (= predict(legacy=False)).

    Read from paths.yaml:derived_root/predictions if present, else computed and written there. `model`
    may be a loaded model or a callable returning one (only called on a cache miss)."""
    ckpt = best_checkpoint(run_dir)
    out = cache_path(run_dir, ds_path)
    if out.exists():
        c = torch.load(out)
        if c["ckpt"] == ckpt.name:
            return c["pred"].float(), c["lab"].float()
    if callable(model) and not isinstance(model, torch.nn.Module):
        model = model()
    model = model if model is not None else load_model(ckpt)
    model.eval().to(DEVICE)
    binary = run_config(run_dir)["data"]["binary"]
    xs, lab = _inputs(ds_path, binary, model.hparams.corr_input)
    outs = []
    for X in xs:
        X = [x.to(DEVICE) for x in X] if isinstance(X, (list, tuple)) else X.to(DEVICE)
        o = model.model(X)
        outs.append(torch.sigmoid(o[0] if model.regression_head else o).cpu())
    model.cpu()
    pred = torch.cat(outs)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ckpt": ckpt.name, "pred": pred, "lab": lab.bool() if binary else lab}, out)
    return pred, lab


def score_split(run_dir, ds_path, model=None):
    """metrics() of a run on a split, from the prediction cache."""
    pred, lab = cached_predict(run_dir, ds_path, model)
    return metrics(pred, lab, entry_mask(ds_path, lab.shape))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def number(x):
    """Three decimals without the leading zero: 0.9501 -> .950, 0.99996 -> 1.00, -0.012 -> $-$.012."""
    if round(x, 3) < 0:
        return "$-$" + number(-x)
    s = f"{abs(x):.3f}"
    return "1.00" if s == "1.000" else s.removeprefix("0")
