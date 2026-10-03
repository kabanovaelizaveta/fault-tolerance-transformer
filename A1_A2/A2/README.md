# Experiment A2: CIFAR-10 Image Classification

This directory collects the saved results for Experiment A2 in *Fault Tolerance in Transformers*. The experiment evaluates the effects of attention-head failures on image classification using CIFAR-10, including single-head calibration, random multi-head evaluation, dependence analysis, and empirical safe regions.

## Directory structure

```text
A2/
├── README.md
├── notebooks/           # Reserved for analysis notebooks; currently empty
├── configs/             # Saved model configuration
├── results/
│   ├── summary.json
│   ├── tables/
│   └── figures/
├── data/                # Links to source data
├── checkpoints/         # Links to model checkpoints
└── archive/             # Intermediate outputs and alternative exports
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

`archive/intermediate/` retains progress exports and per-pair copula progress files. These files document intermediate computation and are kept separately from the main evaluation export.

A second collection was found under `data/cifar10/` in the source materials. Its result tables, figures, summary, and notes are preserved in `archive/alternative_data_cifar10/`. Some same-named files differ in size from those under `results/cifar10/`; the collections have therefore not been merged or treated as verified duplicates. The primary directory layout follows `results/cifar10/` for provenance, not as a claim that this version is scientifically authoritative.

One figure named `experiment_A1_agnews_single_head_four_panels.png` was found in the CIFAR-10 results. It is retained in `archive/needs_review/` because the dataset label in its name conflicts with its source location. Its content has not been reassigned to AG News. Original source notes are preserved in `archive/source_notes/`.

## Data, code, and reproducibility

Dataset and checkpoint links are provided in [data/README.md](data/README.md) and [checkpoints/README.md](checkpoints/README.md). Shared Python modules are stored once in [../src/faulttol/](../src/faulttol/).

The `notebooks/` directory is intentionally empty. No notebook was copied into this package. The saved source code and summaries may refer to the original directory layout; execution paths and environment requirements must be configured before running analyses from this layout.

This package preserves saved artifacts without changing their numerical contents. Organization and file presence have been checked, but training, inference, bootstrap calculations, table-to-figure consistency, and agreement with the manuscript have not been independently revalidated as part of this packaging step. Copula files retain their original grouping; the pending reconciliation identified for Experiment B is not assumed to apply to these experiments.

