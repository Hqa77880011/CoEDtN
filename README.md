# CoEDtN

Implementation of composable energy-based Dirichlet-to-Neumann operators for variable-topology thermal simulation.

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

`prepare` generates a finite-volume resistor-network conduction corpus with four corner ports per component. All boundary excitations of one parent geometry remain in one partition. Arrays contain geometry, power, port temperature/flux, dense fields, boundary lifts, parent IDs and split. `config.json` fixes learning and assembly settings. This source is generated conduction data.

```bash
python -m coedtn prepare --config config.json --data data/components.npz --parents 300 --excitations 8 --seed 11
```

## Training and inference

```bash
python -m coedtn train --config config.json --data data/components.npz --output runs/energy --variant energy --seed 11
```

## Evaluation

```bash
python -m coedtn evaluate --config config.json --model runs/energy/model.pt --output runs/energy/evaluation --seed 11
```

## Ablations

Supported controls: `direct`, `unshared`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
python -m coedtn train --config config.json --data data/components.npz --output outputs/direct --variant direct --seed 11
```

## Outputs

Training writes `model.pt` and `training.json`. Evaluation writes field NPZ arrays, `metrics.json` and `topology.png`.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
