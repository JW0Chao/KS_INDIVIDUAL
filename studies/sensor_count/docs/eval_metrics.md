# Evaluation Metrics in `evaluation.py`

This file explains what each metric means and how it is computed in `evaluation.py`.

## 1) How Dwell Time Is Obtained

`dwell_time` is provided by you through CLI:

```bash
--dwell-time <seconds>
```

The script converts it to discrete simulation steps:

`m = ceil(dwell_time / dt)`

Where:
- `dt` comes from `KS.py`.
- `dt` is checked against `--dt-expected` (default `0.05`).

Example:
- if `dwell_time = 1.0 s` and `dt = 0.05 s`, then `m = ceil(1.0 / 0.05) = 20` steps.

## 2) Instantaneous Control Error

At each rollout step `t_n`, the script computes:

`e_k(t_n) = ||v_k(t_n) - E3||_2`

Implementation:
- full-state L2 norm: `np.linalg.norm(obs - u_target)`.

## 3) Epsilon (Stabilization Threshold)

The uniform study uses:

- target-relative: `epsilon = beta * ||E3||_2`

Controls:
- `--epsilon-beta`
- `--epsilon-mode target_relative`

## 4) Metric 1: Final Control Error

For each rollout `k`, a final window is selected from:
- `--final-window-steps`, or
- `--final-window-frac` (default last 20%).

Then:

`e_final_bar(k) = mean_{n in final_window} e_k(t_n)`

Reported per model:
- mean of `e_final_bar(k)`
- sample std
- median

Related plots:
- `error_mean_vs_time.png` (mean curve only, one line per model)
- `error_median_vs_time.png` (median curve only, one line per model)
- `plot_1_1_error_vs_time.png` (backward-compatible alias to `error_mean_vs_time.png`)
- `plot_1_2_final_error_boxplot.png`

Interactive editing (no rerun needed):
- if `--error-plot-engine plotly`, HTML companions are written:
  - `error_mean_vs_time.html`
  - `error_median_vs_time.html`
- these support interactive edits such as legend movement and text edits for title/axis labels.

## 5) Metric 2: Stabilization Success and Time-to-Stabilize

Success condition for rollout `k`:
- there exists index `n*` such that:
- `e_k(t_j) <= epsilon` for all `j in [n*, n* + m - 1]`

where `m` is dwell steps from Section 1.

Time-to-stabilize:
- `t_stab(k) = n* * dt` if success
- `t_stab(k) = NaN` if not successful

Reported per model:
- success rate
- mean `t_stab` among successful runs
- median `t_stab` among successful runs

Related plots:
- `plot_2_1_success_rate.png`
- `plot_2_2_tstab_boxplot_success_only.png`

## 6) Fairness Rule Across Models

All models use the exact same sampled rows from `INIT.dat` in one run.

This guarantees fair comparison across the evaluated sensor-count models and any optional full-state reference.

## 7) Output Files

Each evaluation run writes:
- `run_config.json`
- `metrics_summary.json`
- `metrics_arrays.npz`
- plot files listed above

## 8) Split Error Plot Controls

You can customize the clean mean/median error plots using:

- `--error-plot-engine {plotly,matplotlib}`
- `--error-show-epsilon` / `--error-hide-epsilon`
- `--error-plot-width`
- `--error-plot-height`
- `--error-line-width`
- `--error-template` (Plotly template)
- `--error-x-label`
- `--error-y-label`
- `--error-title-mean`
- `--error-title-median`

## 9) Stabilization Plots

The run writes target-relative stabilization figures, including:

- `plot_2_1_success_rate.png`
- `plot_2_2_tstab_boxplot_success_only.png`
- `plot_2_4_tstab_diminishing_returns_vs_sensor_count.png`
