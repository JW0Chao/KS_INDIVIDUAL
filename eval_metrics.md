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

## 2) Instantaneous Tracking Error

At each rollout step `t_n`, the script computes:

`e_k(t_n) = ||v_k(t_n) - E3||_2`

Implementation:
- full-state L2 norm: `np.linalg.norm(obs - u_target)`.

## 3) Epsilon (Stabilization Threshold)

The script uses Method 3 (target-relative threshold):

`epsilon = beta * ||E3||_2`

with:
- `beta` from `--epsilon-beta` (default `0.10`)
- `E3` from `--target-file` (default `u3.dat`)

## 4) Metric 1: Final Tracking Error

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
- `plot_1_1_error_vs_time.png` (mean +/- std vs time)
- `plot_1_2_final_error_boxplot.png`

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

## 6) Metric 3: Control Effort

Instantaneous action magnitude:

`||u_k(t_n)||_2^2`

Integrated effort per rollout:

`J_u(k) = sum_{n=0..M} ||u_k(t_n)||_2^2 * dt`

Reported per model:
- mean `J_u`
- sample std
- median
- mean normalized effort `J_u / T`

Related plots:
- `plot_3_1_effort_vs_final_error.png`
- `plot_3_2_effort_boxplot.png`

## 7) Paired SHAP vs Bucci Plot

If both roles exist (`shap8` and `bucci8`), the script computes:

`Delta_k = e_final_bar_SHAP(k) - e_final_bar_Bucci(k)`

using the same initial-condition row `k` for both models.

Output:
- `plot_final_error_paired_diff.png`

Interpretation:
- `Delta_k < 0`: SHAP is better for that rollout.

## 8) Fairness Rule Across Models

All models use the exact same sampled rows from `INIT.dat` in one run.

This guarantees fair comparison across Bucci, SHAP, and optional Full-64 reference.

## 9) Output Files

Each evaluation run writes:
- `run_config.json`
- `metrics_summary.json`
- `metrics_arrays.npz`
- plot files listed above
