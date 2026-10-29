# Provenance (read-only)

Notebooks copied verbatim from the original working repo (`cp_legacy_artifacts/Causal_pretraining`,
state of Feb 2024). They are **not** meant to be run (paths point to the old machines); they
document how the numbers in the paper and the pre-2026 thesis draft were produced.

| Notebook | Documents |
|---|---|
| `data_gen.ipynb` | `make_deterministic_ds` calls and the `joint` merge behind the original datasets. The notebook was edited in place, so not every set's call survives; the authoritative generation arguments are each dataset's `args.p` in the store |
| `prep_kuramoto.ipynb` | conversion of the ACD Kuramoto simulation (`kuramoto5`) into the `kuramoto` set |
| `exp_summarize_results.ipynb` | aggregation of grid and std CSVs into the old result tables |
| `exp_style_result_tables.ipynb` | LaTeX formatting of the old tables and the old scaling figure |

Known issues of the old aggregation, fixed by the `thesis/` scripts:
- the scaling figure takes the max over architectures of per-config means of `Test1_Auroc`
  (= thesis Test-Set 2) and, for the deep/LCM sizes, the grid **validation** max
  (`scores/slurm_exp3_pre_max.csv`), so it mixes aggregations and splits;
- all scores come from train-mode scoring (see `../README.md`, "Scoring: legacy vs eval").
