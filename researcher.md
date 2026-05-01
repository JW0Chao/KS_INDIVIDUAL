# Researcher Workflow Guide

This document describes the reproducible uniform-sensor workflow in this repository. It is written for researchers who need to rerun the complete study, inspect the generated artifacts, and understand which stages must be rerun after a methodological change.

The guide is intentionally limited to the uniform sensor-count study.

## 1. Study Scope

The active study asks how the number of uniformly placed sensors affects reinforcement-learning control of the 1-D Kuramoto-Sivashinsky (KS) system.

The controlled experiment is:

- target state fixed to `data/u3.dat`
- initial-condition bank fixed to `data/INIT.dat`
- actuator geometry fixed to 4 equispaced actuators
- sensor counts fixed to `k = 4, 8, 12, 16, 20`
- sensor layouts generated uniformly inside the safe corridors between adjacent actuators
- one shared non-overlapping split from `INIT.dat`
- 3 independent RL seeds per sensor count: `0, 1, 2`
- final comparison based on held-out test performance

To keep the study scientifically fair, do not compare runs that were produced with different split manifests, different layout logic, different targets, or different training budgets.

## 2. Repo Map

Core inputs:

- `data/INIT.dat`: full-state initial-condition bank used for train, validation, and held-out test rows
- `data/u3.dat`: target state for the study
- `data/x.dat`: spatial grid

Main runnable entrypoints:

- `scripts/make_init_split.py`: create or validate the shared `INIT.dat` split manifest
- `scripts/generate_placements.py`: generate the uniform layout family and placement plots
- `scripts/run_placement_campaign.py`: expand a placement manifest into per-layout, per-seed training runs
- `scripts/train_run.py`: train one RL controller for one layout and one seed
- `scripts/evaluate_sensor_count.py`: evaluate trained controllers on the shared held-out split
- `scripts/plot_placement.py`: render sensor and actuator locations

Reusable implementation code:

- `src/ks_control/ks.py`: KS simulator
- `src/ks_control/model.py`: actor and critic networks
- `src/ks_control/training.py`: DDPG trainer
- `src/ks_control/study_protocol.py`: split loading, validation checkpoint rule, and run/setup aggregation helpers
- `src/ks_control/placement_helpers.py`: uniform placement construction and validation helpers
- `src/ks_control/evaluation_helpers.py`: evaluator checkpoint and sensor-index helpers

Study folders:

- `studies/controller_protocol/manifests/`: shared split manifest
- `studies/sensor_placement/layouts/`: tracked uniform layout JSON files
- `studies/sensor_placement/manifests/`: tracked placement manifests
- `studies/sensor_placement/results/`: generated placement plots
- `studies/sensor_count/results/`: generated evaluation outputs
- `artifacts/raw/sensor_placement/training_bundle/`: generated controller runs, campaign manifest, and evaluation spec

Verification:

- `tests/`: protocol and helper regression tests for the reproducible uniform workflow

## 3. Scientific Contract

### Physical system

- PDE: 1-D Kuramoto-Sivashinsky system on `[0, L]` with `L = 22`
- spatial resolution: `N = 64`
- simulator timestep: `dt = 0.05`
- actuators: 4 fixed Gaussian-shaped actuators

### Sensor-placement variable

- independent variable: number of sensors `k`
- admissible sensor counts in the tracked uniform study: `4, 8, 12, 16, 20`
- placement rule: divide sensors evenly across the 4 safe corridors between adjacent actuators
- safety rule: exclude the actuator index and the `+/- 2` neighboring grid points around each actuator
- implementation: `build_uniform_indices(...)` in `src/ks_control/placement_helpers.py`

The tracked sensor counts are all divisible by 4, which is required by the current uniform placement helper.

### Shared split from `INIT.dat`

The study uses one fixed non-overlapping split manifest:

- train rows: 20
- validation rows: 20
- test rows: 30
- split seed: `123`

The same split must be reused across all sensor counts and all RL seeds.

### RL training protocol

- algorithm: DDPG with replay buffer and parameter noise
- controller target: `u3.dat`
- training seeds: `0, 1, 2`
- training budget: `1000` episodes, `5000` simulator steps per episode
- episode resets: sampled from the shared `train_rows`
- validation: every `25` episodes on all `val_rows` with exploration disabled
- validation stabilization threshold: target-relative epsilon, `epsilon = beta * ||u_target||_2`, with `beta = 0.10`
- checkpoint rule:
  - highest validation success rate
  - then lowest mean final error
  - then lowest mean control effort
- held-out evaluation: run the selected checkpoint on all `test_rows`
- held-out evaluation stabilization threshold: target-relative epsilon, `epsilon = beta * ||u_target||_2`, with `beta = 0.10`

## 4. End-To-End Workflow

The full workflow is:

1. create or validate the shared split manifest
2. generate the uniform layout family
3. train the campaign over `k x seed`
4. evaluate the trained bundle on the shared held-out split
5. analyze the generated CSV and JSON outputs

The key boundary is:

- placement scripts decide where the sensors are
- RL scripts decide how the controller acts for a fixed sensor layout

Do not blur those two stages when interpreting results.

## 5. Reproducing The Uniform Study

Use the Python interpreter from your project environment. If you keep a local virtual environment in `.venv`, activate it first or substitute the full interpreter path.

### Step 1: Create or validate the shared split

```powershell
python scripts/make_init_split.py `
  --init-file data/INIT.dat `
  --out studies/controller_protocol/manifests/controller_setup_split.json `
  --split-seed 123 `
  --train-size 20 `
  --val-size 20 `
  --test-size 30
```

Expected result:

- `studies/controller_protocol/manifests/controller_setup_split.json`

Do not regenerate this with different row counts or a different seed if you want results to remain comparable to existing uniform runs.

### Step 2: Generate the uniform layouts

```powershell
python scripts/generate_placements.py --no-show
```

Expected tracked outputs that are refreshed or validated:

- `studies/sensor_placement/layouts/uniform_k4.json`
- `studies/sensor_placement/layouts/uniform_k8.json`
- `studies/sensor_placement/layouts/uniform_k12.json`
- `studies/sensor_placement/layouts/uniform_k16.json`
- `studies/sensor_placement/layouts/uniform_k20.json`
- `studies/sensor_placement/manifests/generated_layouts.json`

Expected generated outputs:

- `studies/sensor_placement/results/placement_plots/`

### Step 3: Train the campaign

```powershell
python scripts/run_placement_campaign.py `
  --placements-manifest studies/sensor_placement/manifests/generated_layouts.json `
  --bundle-root artifacts/raw/sensor_placement/training_bundle `
  --init-split-file studies/controller_protocol/manifests/controller_setup_split.json `
  --train-seeds 0,1,2 `
  --max-episodes 1000 `
  --device cpu
```

Use `--device cuda` on a machine with a working CUDA runtime. Use `--resume` if you want the script to skip runs already marked successful in `run_status.json`.

The campaign script launches one `train_run.py` process per `(layout, seed)` pair and writes:

- `artifacts/raw/sensor_placement/training_bundle/campaign_manifest.json`
- `artifacts/raw/sensor_placement/training_bundle/models_spec_for_evaluation.json`

It also creates one run folder per trained controller under:

- `artifacts/raw/sensor_placement/training_bundle/runs/<setup_name>/<layout_name>/<target_name>/seed_<seed>/`

### Step 4: Evaluate the trained bundle

```powershell
python scripts/evaluate_sensor_count.py `
  --models-spec artifacts/raw/sensor_placement/training_bundle/models_spec_for_evaluation.json `
  --init-split-file studies/controller_protocol/manifests/controller_setup_split.json `
  --split-role test `
  --dwell-time 1.0 `
  --epsilon-mode target_relative `
  --no-show
```

By default the evaluator writes to:

- `studies/sensor_count/results/evaluation/`

For named experiment outputs, pass `--outdir studies/sensor_count/results/<experiment_name>`.

### Optional: Debug one controller directly

If you need to debug a single layout outside the campaign wrapper, use `train_run.py` directly:

```powershell
python scripts/train_run.py `
  --exp-name uniform_k8__u3__seed_0 `
  --setup-name uniform_k8 `
  --sensor-indices-file studies/sensor_placement/layouts/uniform_k8.json `
  --state-dim 8 `
  --init-split-file studies/controller_protocol/manifests/controller_setup_split.json `
  --train-seed 0 `
  --device cpu
```

This is useful for debugging, but the main study should be launched through `run_placement_campaign.py` so the bundle structure and evaluation spec stay consistent.

## 6. What Each Stage Produces

### Split stage

Tracked artifact:

- `studies/controller_protocol/manifests/controller_setup_split.json`

Critical fields:

- `split_seed`
- `train_rows`
- `val_rows`
- `test_rows`
- `init_file`

### Placement stage

Tracked artifacts:

- `studies/sensor_placement/layouts/uniform_k*.json`
- `studies/sensor_placement/manifests/generated_layouts.json`
- `studies/sensor_placement/manifests/baseline_uniform_layouts.json`

Generated artifacts:

- `studies/sensor_placement/results/placement_plots/*.png`
- `studies/sensor_placement/results/placement_plots/*.pdf`
- `studies/sensor_placement/results/placement_plots/*.svg`

### Training stage

Per-run generated artifacts inside each run directory:

- `run_config.json`
- `run_status.json`
- `train.log`
- `sensor_indices.json`
- `model/validation_rows.json`
- `model/validation_history.csv`
- `model/best_checkpoint_meta.json`
- numeric actor and critic checkpoints such as `250_actor.pt`

The most important status file is `run_status.json`. A successful run records:

- selected checkpoint episode
- best validation metrics
- split and seed provenance
- paths to the validation metadata

### Evaluation stage

Generated outputs in the evaluation outdir:

- `run_config.json`
- `metrics_summary.json`
- `test_summary.json`
- `test_per_row.csv`
- `test_run_summary.csv`
- `test_setup_summary.csv`
- `metrics_arrays.npz`
- summary plots

Interpret them as follows:

- `test_per_row.csv`: one row per held-out initial condition
- `test_run_summary.csv`: one row per trained controller run
- `test_setup_summary.csv`: one row per sensor count after aggregating across seeds

## 7. How To Interpret The Comparison

The main comparison object is the sensor-count setup, not a single lucky controller checkpoint.

Aggregation happens in two levels:

1. aggregate held-out rows within each trained run
2. aggregate run summaries across the 3 RL seeds for each sensor count

The setup ranking rule is:

1. highest mean success rate
2. then lowest mean final error
3. then lowest mean control effort

This ranking should be decided before inspecting results and kept fixed for the whole study.

## 8. What Must Be Rerun After A Change

If you change the split manifest:

- rerun all controller training and all evaluation

If you change the uniform placement logic:

- regenerate the layouts
- retrain all affected controllers
- rerun held-out evaluation

If you change the RL training protocol, reward, network, or checkpoint rule:

- retrain all affected controllers
- rerun held-out evaluation

If you change only the evaluator plots or downstream summaries:

- reevaluation is usually sufficient
- retraining is not required unless the metric definition changed

## 9. Common Failure Modes

- Comparing runs produced with different split manifests. Check `run_config.json` and `run_status.json`.
- Editing layout JSON files manually. Prefer `generate_placements.py`.
- Mixing targets within one comparison set. Keep `u3.dat` fixed unless the target itself is the variable under study.
- Treating generated runtime folders as source files. The source of truth for method logic is in `scripts/` and `src/ks_control/`.
- Assuming the latest numeric checkpoint is always the selected checkpoint. Use `best_checkpoint_meta.json` or the `selected_episode` recorded in `run_status.json`.

## 10. Verification Before Long Runs

Before launching a long campaign, run:

```powershell
python -m unittest discover -s tests
python scripts/train_run.py --help
python scripts/run_placement_campaign.py --help
python scripts/evaluate_sensor_count.py --help
python scripts/generate_placements.py --help
```

For a fast smoke test, run one seed on one layout with a tiny episode budget into a temporary bundle root, then evaluate that temporary bundle. Remove the temporary output folder afterward so the worktree stays clean.

## 11. Bottom Line

To reproduce the uniform study faithfully, keep four things fixed:

- the shared `INIT.dat` split
- the uniform corridor-based layout rule
- the 3-seed RL protocol
- the held-out evaluator and ranking rule

If one of those changes, treat the result as a new experiment rather than an extension of the old one.
