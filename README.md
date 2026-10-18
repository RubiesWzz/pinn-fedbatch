# Physics-Informed Neural Network for Fed-Batch Bioprocess Modelling

MATLAB research code for data-driven and physics-informed prediction of fed-batch bioprocess dynamics, accompanied by a logistic fed-batch baseline and parameter sensitivity analysis.

## Overview

The main PINN predicts five states: biomass concentration (**X**), substrate concentration (**S**), total protein (**Pt**), secreted protein (**Pm**), and reactor volume (**V**). The PINN combines supervised data fitting with residuals from a five-equation ODE model. The current main version includes Min-Max normalization, training/validation/test separation by batch, a staged physics-loss schedule, and model evaluation.

This repository also contains a **separate simplified logistic fed-batch model** for exploring parameter estimation and identifiability of `mu_max` and `X_max`. It should not be confused with the PINN's five-state governing equations.

## Repository layout

```text
pinn-fedbatch/
├── run_pinn.m                   # Convenient PINN entry point
├── src/
│   ├── main_pinn.m              # PINN_v12, with dataset path updated
│   ├── feed_rate.m
│   ├── logistic_fedbatch_ode.m
│   ├── simulate_and_sample.m
│   ├── compute_sensitivity_energy.m
│   ├── residual_X.m
│   └── residual_mu_only.m
├── analysis/
│   ├── phase1_main_logistic_fedbatch.m
│   ├── phase1_constant_x_max.m
│   ├── phase2_sensitivity_fd.m
│   └── phase2_compare_inputs.m
├── archive/
│   ├── PINN_v12.m               
│   └── PINN_v9.m                
├── data/                         
├── presentation/                 
└── results/                   
```

## Requirements

- MATLAB with **Deep Learning Toolbox** for the PINN (`dlnetwork`, `dlarray`, `dlfeval`, `adamupdate`).
- MATLAB with **Optimization Toolbox** for the Phase 1 `lsqnonlin` experiments.
- A MATLAB version supporting script-local functions and the language features used in the scripts.
- The input dataset `data/data.csv` (not supplied here).

Toolboxes and code have **not** been executed or verified in MATLAB as part of this repository packaging.

## Getting started

1. Clone or download this repository.
2. Place the authorized input file at **`data/data.csv`** (see `data/README.md` for the expected variables).
3. In MATLAB, set the current folder to the repository root, then run:

```matlab
run_pinn
```


