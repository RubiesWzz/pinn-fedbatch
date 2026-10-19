# Fed-Batch PINN — V15

MATLAB research implementation of a physics-informed neural network (PINN) for a fed-batch bioreactor, with an ODE-only reference model and experimental scripts. **V15 is the default version; V12–V14 are preserved for research history.**

## Repository layout

```text
.
├── run_pinn.m                 # V15 launcher
├── run_ode_baseline.m         # ODE baseline launcher
├── src/                       # V15 PINN and ODE baseline
├── archive/                   # V12, V13 and V14 (not auto-executed)
├── data/                      # Main CSV and supporting CSV datasets
├── analysis/                  # Dataset checks
├── experiments/
│   ├── identifiability/      # Logistic ODE fitting and sensitivities
│   ├── hyperparameters/      # Lambda sweep and stored values
│   └── alternatives/         # Bayesian / weighted-loss exploratory variants
└── present/                   # Place verified figures here
```

## Getting started

1. Open MATLAB and set **Current Folder** to the repository root.
2. Run `run_pinn` for V15, or `run_ode_baseline` for the physics-only reference.
3. Optional data sanity check: `run('analysis/check_project_data.m')`.

The V15 script is configured for up to **50,000 epochs** and requires MATLAB's Deep Learning Toolbox. It is computationally intensive. The supplied source code has not been executed or benchmarked in this packaging environment.

## Dataset and model

- The main dataset is `data/data.csv` and contains 16 batches.
- V15 holds out batch 16 as a test batch; its code trains on batches 1–15 and has **no dedicated validation batches**. Earlier versions use different splits. Do not compare their results as if evaluated with identical protocols.
- Network outputs are biomass X, substrate S, total protein Pt, and secreted protein Pm. Volume V is numerically obtained from the feed profile and used as an input, rather than predicted.
- The Park–Ramirez kinetic ODE is implemented as a physics residual; `src/Pureode_baseline.m` solves the ODE directly.

## Research / evaluation caveats

- V15 periodically computes **test** Pm R² and selects the best network using that metric. Thus reported performance on batch 16 is **not an unbiased held-out test estimate**; a truly independent test should be reserved for final evaluation.
- In V15, the dense collocation pool includes all batches, including batch 16, so physics constraints also access the held-out batch's feed profile / initial conditions. This is a transductive or known-input setup, not a fully untouched test scenario.
- `V_exact` is calculated by numerical integration from recorded feed rates; the printed R²=1 for volume is not a learned neural-network prediction.
- `experiments/alternatives/` contains entire research variants, not importable helper functions. They are retained for reproducibility, but not loaded by `run_pinn`.
- The parameter-identifiability studies use a separate logistic model, not the full Park–Ramirez model; their sensitivity findings should not be conflated with results for the final bioreactor model.
- The exploratory scripts are archived as originally implemented other than relative data paths / function file naming. Some may need additional experimental configuration or debugging for reproduction.
- No trained model weights or verified numeric R² results are provided in this package.

## Dependencies

MATLAB (Deep Learning Toolbox for PINN), and Optimization Toolbox for `lsqnonlin` parameter-fitting studies. Tool availability and results depend on MATLAB version.

