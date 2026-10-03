# Experiment B: Transformer Fault Tolerance in Forecasting

This directory contains the result tables and configuration files for the forecasting experiment in *Fault Tolerance in Transformers*. The experiment examines how failures of individual attention heads and combinations of heads affect prediction error, and evaluates empirical safe regions under specified error thresholds.

Start with `results/tables/` for the primary data and summaries. Copula results awaiting reconciliation are stored separately in `results/copulas_pending/`. Earlier and intermediate outputs are retained in `archive/` for traceability.

## Directory structure

```text
.
├── README.md
├── configs/
│   └── random_multihead_config_regression.json
├── results/
│   ├── tables/
│   └── copulas_pending/
└── archive/
    ├── outdated/
    ├── intermediate/
    └── duplicate_configs/
```

## Primary data and summaries

The primary multi-head dataset is:

[`base_transformer_regressor_safe_region_input_with_mape_and_critical_heads.csv`](results/tables/base_transformer_regressor_safe_region_input_with_mape_and_critical_heads.csv)

It contains 10,000 recorded fault scenarios, including the original multi-head results, MAPE measurements, and critical-head indicators. Use this file as the starting point for analyses of combined head failures.

The remaining files in `results/tables/` provide supporting measurements and summaries:

| File | Contents |
|---|---|
| `head_summary_single_head_regression.csv` | Per-head summaries of single-head fault effects. |
| `damage_matrix_single_head_regression.csv` | Sample-level changes supporting the single-head summaries. |
| `layer_summary_single_head_regression.csv` | Aggregated single-head results by layer. |
| `base_transformer_regressor_head_weights_for_random_faults_regression.csv` | Head weights used to calculate aggregate fault severity. |
| `base_transformer_regressor_single_head_mape_criticality.csv` | MAPE-based head criticality used in the safe-region analysis. |
| `preliminary_correlations_regression.csv` | Correlation results consistent with the primary multi-head dataset. |
| `base_transformer_regressor_clean_test_metrics.csv` | Baseline test metrics without injected faults. |
| `clean_metrics_single_head_regression.csv` | Baseline metrics recorded for the single-head evaluation. |
| `base_transformer_regressor_training_history.csv` | Recorded training history. |

The two baseline-metric files document separate evaluation stages and are retained individually. MAPE-based criticality and the single-head error-change summaries use different evaluation definitions and should be interpreted accordingly.

The random multi-head experiment configuration is available in [`configs/random_multihead_config_regression.json`](configs/random_multihead_config_regression.json).

## Copula analysis status

`results/copulas_pending/` preserves outputs from two fitting workflows:

| Workflow | Fit results | Bootstrap results |
|---|---|---|
| Three-family comparison | `expB_copula_fit_results.csv` | `copula_cvm_bootstrap_results.csv` |
| Five-family comparison | `survival_copula_fit_results.csv` | `survival_cvm_bootstrap_W_fail_vs_delta_mae.csv` |

These files are retained to document the analysis history. Reconciliation of the final copula table is pending. Fitted parameters, goodness-of-fit statistics, and bootstrap results should be taken from a consistent workflow; the five-family bootstrap file covers only the `W_fail`–`delta_mae` pair.

## Archived outputs

| Directory | Purpose |
|---|---|
| `archive/outdated/` | Earlier correlation and empirical tail-validation tables that do not match the primary dataset. These are excluded from the current result set. |
| `archive/intermediate/` | Intermediate multi-head exports and compact summaries whose information is represented in the primary tables. |
| `archive/duplicate_configs/` | An additional copy of the experiment configuration under its original filename. |

Archived files are provided for provenance and compatibility with earlier analysis steps. Use `results/tables/` for current descriptive results and observe the status notes for the copula analysis.

## Scope and reproducibility

This directory contains 23 CSV and JSON files. It is a results-and-configuration collection; the analysis notebook, figures, source dataset, and model checkpoints are not included in this package.

The original notebook references some intermediate files by their earlier paths. Those paths must be updated, or the expected intermediate exports regenerated, before running the notebook against this directory layout. Reorganizing the files does not constitute a new execution of the experiment or its bootstrap analysis.
