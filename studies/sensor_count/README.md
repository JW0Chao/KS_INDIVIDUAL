# Sensor Count Study

## Purpose

This study folder captures the reproducible workflow for measuring how sensor count affects reinforcement-learning stabilization performance for the 1-D Kuramoto-Sivashinsky system.

## Scope

- fixed actuator placement
- uniform sensor placement
- sensor counts `k = 4, 8, 12, 16, 20`
- metrics:
  - control error vs time
  - success rate
  - time-to-stabilize

## Key Files

- evaluation metric notes:
  - `studies/sensor_count/docs/eval_metrics.md`
- generated evaluation outputs:
  - `studies/sensor_count/results/`
- campaign-generated models spec:
  - `artifacts/raw/sensor_placement/training_bundle/models_spec_for_evaluation.json`

## Useful Commands

Evaluate a model set:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_sensor_count.py `
  --models-spec artifacts/raw/sensor_placement/training_bundle/models_spec_for_evaluation.json `
  --dwell-time 1.0 `
  --no-show
```

Generate the rollout triplet figure:

```powershell
.\.venv\Scripts\python.exe scripts/plot_rollout_triplet.py `
  --models-spec artifacts/raw/sensor_placement/training_bundle/models_spec_for_evaluation.json
```

All files under `studies/sensor_count/results/` are generated outputs and are intentionally not tracked in this branch.
