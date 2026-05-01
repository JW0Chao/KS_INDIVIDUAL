# Sensor Placement Study

## Purpose

This study folder contains the baseline uniform placement workflow used to launch comparable controller runs from fixed sensor layouts.

## Current Baseline

- system size: `N = 64`
- actuator count: `4`
- baseline placement family: uniform layouts
- baseline sensor counts: `k = 4, 8, 12, 16, 20`

The live workflow in this branch is baseline-first. Any future non-uniform strategy should be added explicitly as a new layout family rather than mixed into the baseline assets.

## Folder Layout

- layouts:
  - `studies/sensor_placement/layouts/`
- manifests:
  - `studies/sensor_placement/manifests/`
- generated placement figures:
  - `studies/sensor_placement/results/`
- training bundle:
  - `artifacts/raw/sensor_placement/training_bundle/`

## Immediate Next Experiments

- compare non-uniform placements against the uniform baseline
- test robustness under noisy measurements
- later compare RL performance with classical model-based baselines under identical sensing

## Useful Commands

Create or validate the shared train/val/test split first:

```powershell
.\.venv\Scripts\python.exe scripts/make_init_split.py
```

Regenerate the uniform baseline layouts:

```powershell
.\.venv\Scripts\python.exe scripts/generate_placements.py --no-show
```

Train a placement campaign from a manifest:

```powershell
.\.venv\Scripts\python.exe scripts/run_placement_campaign.py `
  --placements-manifest studies/sensor_placement/manifests/generated_layouts.json `
  --max-episodes 1000 `
  --device cpu
```

Evaluate a trained controller bundle against the shared held-out split:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_sensor_count.py `
  --models-spec artifacts/raw/sensor_placement/training_bundle/models_spec_for_evaluation.json `
  --dwell-time 1.0 `
  --no-show
```

The placement figures, training bundle, and evaluation outputs are generated artifacts. They are kept out of git so this branch stays source-clean.
