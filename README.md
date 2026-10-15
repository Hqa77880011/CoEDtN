# CoEDtN

Implementation of composable energy-based Dirichlet-to-Neumann operators for variable-topology thermal simulation.

## Install

Use Python 3.10 or later and install with `python -m pip install -e .`.

## Data and training

```bash
python -m coedtn prepare --parents 300 --excitations 8
python -m coedtn train --variant energy --output outputs/full
python -m coedtn train --variant direct --output outputs/direct
python -m coedtn train --variant unshared --output outputs/unshared
python -m coedtn evaluate --model outputs/full/model.pt --output outputs/topology
python -m coedtn evaluate --model outputs/full/model.pt --component-holdout --output outputs/component
python -m coedtn evaluate --model outputs/direct/model.pt --output outputs/direct-eval
python -m coedtn evaluate --model outputs/unshared/model.pt --output outputs/unshared-eval
```

`prepare` generates a finite-volume resistor-network reference corpus with four corner ports per component. Every parent geometry, source, and all its independent boundary excitations stay in one partition. `g` contains conductivity, horizontal size, vertical size, and family ID. The NPZ also holds power, port temperature `t`, outward flux `q`, dense field, harmonic boundary lift, parent ID, and split. An exported FEM corpus can use this schema after consistent port integration; the bundled generator supplies a concrete discretized conduction problem.

The energy head implements `B.T diag(softplus(w)) B + epsilon Pi`; source projection preserves total power. Port loss trains reusable responses. The decoder receives the solved boundary trace and vanishes at ports. Assembly uses a sparse interface system with Robin anchoring. Direct-flux and unshared heads are independently trained controls. Evaluation also executes no-solve, identity-shuffle and reference-DtN controls under the same assemblies.

`config.json` controls learning rate, batch size, epochs, early stopping, field loss weight, topology sizes, Robin conductance and physical length units. Training writes `model.pt` and validation history. Evaluation writes field arrays, metric records and `topology.png`. Field NRMSE uses the reference temperature range; hotspot distance uses configured millimetres; flux residual includes external Robin exchange and energy residual compares integrated source with outward port flux. Latency comes from the executed inference/assembly path, with a P95 summary across configurations.

## Implementation conventions

The built-in corpus uses a 4-by-4 finite-volume grid and linear steady conduction. Its reference-DtN control is a discretization reference. The family holdout increases conductivity outside the training interval; topology counts are separately configurable. GINO, DAFNO and other full-domain models can be compared by exporting predictions against the same field arrays and partitions; their architectures are not renamed as port-head controls. The paper does not define the normalization of its Gap score, so this repository reports the specified field and conservation errors separately rather than assigning an arbitrary Gap value.
