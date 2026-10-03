# Transformer Fault Tolerance: Classification Experiments

This collection accompanies *Fault Tolerance in Transformers* and organizes the saved materials for two classification experiments:

|Experiment|Dataset|Task|
|-|-|-|
|[A1](A1/README.md)|AG News|Text classification|
|[A2](A2/README.md)|CIFAR-10|Image classification|

 

Each experiment uses the same top-level structure: `configs/`, `results/tables/`, `results/figures/`, `archive/`, and an empty `notebooks/` directory. Dataset and checkpoint references are documented separately. Shared Python modules are stored in `src/faulttol/`; generated Python caches are omitted.

## Scope

The package preserves existing tables, figures, metadata, and shared source code. Notebook files have intentionally been excluded. Large datasets and model checkpoints remain in the source Drive folders and are referenced from each experiment's README files.

For A2, alternative exports originally stored under `data/cifar10/` are retained separately in the archive. Their relationship to the primary `results/cifar10/` exports remains to be reconciled. Packaging does not imply that all saved outputs have been validated against the manuscript or reproduced from a checkpoint.

File contents and original filenames are preserved. Paths used by the original analyses may require adjustment for this directory layout. Empty notebook directories exist on Drive; Git does not track empty directories.

