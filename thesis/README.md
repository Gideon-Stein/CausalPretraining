# Thesis reproduction (CP v1, Ch. 7.1)

This folder reproduces every CP v1 number in the dissertation *On Causal Discovery for
Real-World Applications*. It only adds code: the released model, data and training code
is used as is, so all original checkpoints load and score identically.

## What lives where

| What | Where | In git |
|---|---|---|
| Code, configs, small result CSVs | this repo (`experimental_results/`, `thesis/`) | yes |
| Original datasets + baseline scores | `paths.yaml:store_root` (read-only) | checksums in `manifest/*.md5` |
| Training runs + checkpoints | `paths.yaml:runs_root` (`cp_1`) | mapping in `manifest/runs.yaml` |

`data/generate_synthetic_ds.py` is unseeded, so regenerated datasets differ from the
originals. Always use the store.

## Naming

The result files use the original dataset names and swapped test-set columns.
`common.py` translates both; scripts should never use the raw names directly.

| Thesis | Folder | | Thesis | Stored column |
|---|---|---|---|---|
| SL | `base` | | Test-Set 1 (in-distribution, `<ds>_additional/simple_test.p`) | `Test2_AUROC` |
| ML | `big_base_mirror` | | Test-Set 2 (shifted, `<ds>/simple_test.p`) | `Test1_Auroc` |
| SNL | `poly` | | | |
| MNL | `big_poly` | | | |
| LNL | `nl` | | | |
| XLNL | `big_nl` | | | |

Scaling: 5 vars = `big_base_mirror`, 7 = `scale_up_2`, 10 = `scale_up`, 12 = `scale_up_4`, 15 = `scale_up_3`.

## Scripts

| Script | Produces |
|---|---|
| `00_verify.py` | checks data checksums, run mapping, architectures, and re-scores sample runs |

Run everything from the repo root in the `causal_pretraining` environment, e.g.
`python thesis/00_verify.py`.
