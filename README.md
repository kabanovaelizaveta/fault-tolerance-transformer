# Fault Tolerance in Transformers

Research artifacts accompanying *Fault Tolerance in Transformers*. The experiments investigate how attention-head failures affect Transformer predictions in text classification, image classification, and time-series forecasting.

## Experiments

| Experiment | Task | Materials |
|---|---|---|
| A1 | Text classification on AG News | [Experiment A1](A1_A2/A1/README.md) |
| A2 | Image classification on CIFAR-10 | [Experiment A2](A1_A2/A2/README.md) |
| B | Time-series forecasting | [Experiment B](B_AEP/README.md) |

## Repository layout

```text
.
├── A1_A2/
│   ├── A1/
│   ├── A2/
│   └── src/faulttol/
└── B_AEP/
```

Each experiment README describes its result tables, configuration records, and archived outputs. Classification figures are included under the corresponding `results/figures/` directories. Shared classification code is stored in `A1_A2/src/faulttol/`.

## Using the materials

Start with the README for the relevant experiment. Primary result tables and historical exports are stored separately; consult the status notes before selecting values for analysis or reporting.

Dataset and checkpoint references for A1 and A2 are provided in their `data/README.md` and `checkpoints/README.md` files. Access to linked files depends on the permissions of the source Google Drive collection.

## Reproducibility status

This repository currently provides saved research artifacts and shared classification modules. Analysis notebooks are not yet included, and execution paths and environment requirements still need to be configured for this layout.

Alternative CIFAR-10 exports are retained separately for provenance. 
