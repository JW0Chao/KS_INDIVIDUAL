# DDPG Code Overview

This document explains what each Python script in this folder does and how the full workflow runs.

## High-Level Flow

Training flow:
1. `main.py` sets experiment config and starts the loop.
2. `KS.py` advances the Kuramoto-Sivashinsky system state each step.
3. `train.py` chooses actions and updates actor/critic using replay data.
4. `buffer.py` stores and samples transitions.
5. `model.py` defines the actor and critic neural networks.
6. `utils.py` and `param_noise.py` provide update and exploration utilities.
7. Checkpoints and buffer snapshots are saved to experiment folders.

Evaluation flow:
1. `main.py` with `Test=True` runs policy only (no training), optionally from random rows in `INIT.dat`.
2. `evaluate_metrics.py` runs batch evaluations and generates metric plots.

## Script-by-Script Explanation

## `main.py`

Purpose:
- Main entry script for training or test-mode rollout.
- Defines experiment settings (episodes, sensors, folders, plotting, etc.).

How it works:
- Loads target state `u3.dat` and grid `x.dat`.
- Creates experiment folders using `EXP_NAME`:
  - `MODEL_DIR = ./Model_<EXP_NAME>`
  - `BUFFER_DIR = ./Buffer_<EXP_NAME>`
- Builds:
  - `MemoryBuffer` from `buffer.py`
  - `Trainer` from `train.py`
  - `KS` simulator from `KS.py`
- If `Test=True`, loads `INIT.dat` and picks a random row as initial condition each episode.
- At each step:
  - Samples sensor state from full field.
  - Gets action from trainer.
  - Advances PDE with `ks.advance(...)`.
  - Computes reward as negative deviation to target.
  - Adds transition to replay and optimizes (training mode only).
- Saves checkpoints every 10 episodes (training mode only).

Key switch:
- `Test=False`: training mode.
- `Test=True`: inference/evaluation mode (no replay writes, no optimizer updates).

## `train.py`

Purpose:
- Implements DDPG training logic and model checkpoint management.

How it works:
- `Trainer` builds:
  - online actor/critic
  - target actor/critic
  - perturbed actor for parameter-noise exploration
- `get_action(...)`:
  - uses perturbed actor for exploration in training
  - disables noise in test mode
- `optimize(...)`:
  - samples batch from replay
  - computes critic target: `r + gamma * Q_target(s', a')`
  - updates critic via Smooth L1 loss
  - updates actor via deterministic policy gradient objective
  - softly updates target networks
- `update_pert(...)`:
  - adapts parameter-noise scale using action-distance metric
- `save_models(...)` and `load_models(...)`:
  - save/load actor, critic, target networks
  - save/load replay-related training state from buffer directory

## `model.py`

Purpose:
- Defines neural network architectures used by DDPG.

How it works:
- `Actor`:
  - input: state vector (`state_dim`)
  - output: action vector (`action_dim`)
  - structure: MLP + `tanh`, scaled by `action_lim`
- `Critic`:
  - input: state and action
  - output: scalar Q-value
  - combines processed state and action branches
- Includes weight initialization helpers:
  - `fanin_init`
  - `swish`

## `KS.py`

Purpose:
- Environment/simulator for controlled Kuramoto-Sivashinsky dynamics.

How it works:
- Builds periodic spatial grid and spectral operators.
- Builds actuator basis fields (`B`) using shifted Gaussian profiles.
- `advance(u0, action)`:
  - maps action amplitudes to distributed forcing field
  - advances PDE using semi-implicit RK3 spectral update
  - returns next state in physical space

## `buffer.py`

Purpose:
- Replay buffer storage for off-policy DDPG.

How it works:
- Stores transitions `(state, action, reward, next_state)` in a deque.
- `sample(batch_size)` returns random mini-batches.
- Tracks metadata (`len`, `pos`, `cont`, recent indices).
- `save_buffer(...)` / `load_buffer(...)` serialize replay contents and counters.
- Uses configurable `buffer_dir`, so different experiments can use separate folders.

## `utils.py`

Purpose:
- Small helper utilities for training updates and exploration process.

How it works:
- `soft_update(...)`: Polyak averaging for target networks.
- `hard_update(...)`: direct parameter copy.
- `OrnsteinUhlenbeckActionNoise`:
  - temporally correlated action noise process used in training.

## `param_noise.py`

Purpose:
- Parameter-noise adaptation utilities (from OpenAI Baselines style).

How it works:
- `AdaptiveParamNoiseSpec` tracks current stddev and adapts up/down.
- `ddpg_distance_metric(...)` computes RMS distance between two action sets.
- Used by `Trainer.update_pert(...)` to tune perturbation intensity.

## `evaluate_metrics.py`

Purpose:
- Standalone evaluator for saved checkpoints with multi-run metric plots.

How it works:
- Loads checkpoint from a model folder (specific episode or latest).
- Runs multiple rollouts (`--num-evals`, default 20).
- For each rollout:
  - picks random initial condition row from `INIT.dat`
  - computes:
    - value function vs time (`Q(s,a)`)
    - state deviation vs time (`||u-u_target||`)
    - time to stabilize (threshold + hold window)
- Plots all runs:
  - each run in gray
  - mean in black
  - median in red
- Saves figure to `--output-plot`.

## Typical Usage

Train:
```powershell
.\.venv\Scripts\python.exe main.py
```

Evaluate 4-sensor model:
```powershell
.\.venv\Scripts\python.exe evaluate_metrics.py --models-dir .\Model_4sensors_baseline --s-dim 4 --num-evals 20 --max-steps 3000 --output-plot eval_4sensors_baseline.png
```

Evaluate 64-sensor model:
```powershell
.\.venv\Scripts\python.exe evaluate_metrics.py --models-dir .\Model_64sensors --s-dim 64 --num-evals 20 --max-steps 3000 --output-plot eval_64sensors.png
```
