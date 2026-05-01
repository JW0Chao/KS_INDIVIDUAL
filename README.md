# KS Uniform Sensor Study


The repo is organized around two connected tracks:

- `studies/sensor_count/`
  - the completed study on how sensor count affects stabilization performance
- `studies/sensor_placement/`
  - the baseline uniform-placement workspace used to launch comparable controller runs

The core reproducible pipeline is:

1. create or validate the shared initial-condition split with `scripts/make_init_split.py`
2. generate baseline uniform layouts with `scripts/generate_placements.py`
3. expand a layout manifest into training runs with `scripts/run_placement_campaign.py`
4. train individual controllers with `scripts/train_run.py`
5. evaluate trained controllers with `scripts/evaluate_sensor_count.py`

Generated outputs are recreated on demand and are intentionally not tracked in this branch. The main runtime locations are:

- `artifacts/raw/sensor_placement/training_bundle/`
- `studies/sensor_placement/results/`
- `studies/sensor_count/results/`

This branch intentionally excludes SHAP, observability-aware selection, and other exploratory placement pipelines so the uniform study can be rerun without mixed research context.

## Environment

Use Python 3.12 and install the runtime dependencies before running the study:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The reported protocol uses `--max-episodes 1000`, `--max-steps 5000` during training, and the target-relative stabilization threshold `epsilon = 0.10 * ||u3||_2`. These are the branch defaults.

## Reproduce The Uniform Study

```powershell
.\.venv\Scripts\python.exe scripts/make_init_split.py `
  --init-file data/INIT.dat `
  --out studies/controller_protocol/manifests/controller_setup_split.json `
  --split-seed 123 `
  --train-size 20 `
  --val-size 20 `
  --test-size 30

.\.venv\Scripts\python.exe scripts/generate_placements.py --no-show

.\.venv\Scripts\python.exe scripts/run_placement_campaign.py `
  --placements-manifest studies/sensor_placement/manifests/generated_layouts.json `
  --bundle-root artifacts/raw/sensor_placement/training_bundle `
  --init-split-file studies/controller_protocol/manifests/controller_setup_split.json `
  --train-seeds 0,1,2 `
  --max-episodes 1000 `
  --device cpu

.\.venv\Scripts\python.exe scripts/evaluate_sensor_count.py `
  --models-spec artifacts/raw/sensor_placement/training_bundle/models_spec_for_evaluation.json `
  --init-split-file studies/controller_protocol/manifests/controller_setup_split.json `
  --split-role test `
  --dwell-time 1.0 `
  --epsilon-mode target_relative `
  --no-show
```
