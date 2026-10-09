# Experiment A1: AG News Text Classification

This directory collects the saved results for Experiment A1 in *Fault Tolerance in Transformers*. The experiment evaluates the effects of attention-head failures on text classification using AG News, including single-head calibration, random multi-head evaluation, dependence analysis, and empirical safe regions.

## Directory structure

```text
A1/
├── README.md
├── notebooks/           # Reserved for analysis notebooks; currently empty
├── configs/             # Saved model configuration and vocabulary
├── results/
│   ├── summary.json
│   ├── tables/
│   └── figures/
├── data/                # Links to source data
├── checkpoints/         # Links to model checkpoints
└── archive/             # Intermediate outputs
```

## Where to start

- [Model configuration](configs/paper_table_1_model_configuration.csv) provides the saved configuration table.
- [Experiment summary](results/summary.json) records model and training settings, evaluation summaries, and the original run paths.
- [Multi-head evaluation](results/tables/random_multihead_independent_eval.csv) contains the scenario-level fault-evaluation results.
- [Single-head calibration](results/tables/single_head_damage_calibration.csv), [head weights](results/tables/head_weights_calibration.csv), and [layer summaries](results/tables/layer_summary.csv) describe individual-head vulnerability.
- [Safe-region grid](results/tables/safe_region_grid.csv), [summary](results/tables/safe_region_summary.csv), and [required thresholds](results/tables/required_tau.csv) document the empirical safe-region analysis.

Additional tables contain correlations, copula fits, bootstrap outputs, data-split indices, and training history. Files beginning with `paper_table_` and formatted summaries are presentation exports; their filenames retain the numbering used when they were generated and may differ from the current manuscript. The detailed tables should be consulted alongside these exports.

## Figures and analysis history

`results/figures/` contains the figures saved under the source experiment's results directory. These include alternative presentations and supporting plots; inclusion does not identify a figure as the final manuscript version.

`archive/intermediate/` retains progress exports. These files document intermediate computation and are kept separately from the main evaluation export.

## Data, code, and reproducibility

Dataset and checkpoint links are provided in [data/README.md](data/README.md) and [checkpoints/README.md](checkpoints/README.md). Shared Python modules are stored once in [../src/faulttol/](../src/faulttol/). The saved vocabulary is retained as `configs/vocab.json`.


