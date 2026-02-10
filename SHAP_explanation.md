# SHAP Sensor-Finding Workflow (Current)

This document describes the current SHAP ranking procedure in this repo.

## Goal

Select sensors using a combined importance that reflects:
1. true rollout performance relevance (`J`), and
2. control decision relevance (`Q`).

## Scripts

1. `shap_sensor_find.py`
2. `shap_surrogate.py`
3. `shap_utils.py`
4. `compare_masked_sensors.py`

## Labels Used for SHAP

1. Metric 1 label (`J`): rollout-based cost (`rollout_j`)
- computed by closed-loop KS rollouts
- default metric: mean L2 deviation over horizon

2. Metric 2 label (`Q`): critic action-value (`critic_q`)
- computed as `Q(s, pi(s))`

## SHAP Global Importance per Label

For each label, local SHAP values `phi_i` are aggregated as:

`I_i = mean(abs(phi_i))`

This gives:
1. `I_j` for rollout label
2. `I_q` for critic label

## Final Combined Importance (Default Ranking)

Both per-label importance vectors are normalized first:

`I_j_norm_i = I_j_i / sum(I_j)`

`I_q_norm_i = I_q_i / sum(I_q)`

Then combined ranking is:

`I_combined_i = alpha * I_q_norm_i + (1 - alpha) * I_j_norm_i`

Default:
- `alpha = 0.5`
- equivalent to `0.5 * I_q_norm + 0.5 * I_j_norm`

## Grouping Behavior

1. Auto two-stage (default)
- coarse stage: grouped features (default group size 8)
- fine stage: sensor-level within top coarse groups

2. Combined importance is used for selection in both stages:
- coarse top-group selection
- fine sensor ranking and top-k selection

## Top-k Files

At sensor-level stage (`group_size == 1`):
1. `topk_raw.json`
- first `k` sensors from combined ranking

2. `topk_actuator_excluded.json`
- actuator-constrained selection from combined ranking
- excludes sensors in a window around each actuator center before selecting top-k

3. `topk_actuator_excluded_meta.json`
- includes actuator indices, exclusion-window settings, and ranking source (`combined`)

Backward-compatible aliases are still written:
1. `topk_spacing.json`
2. `topk_spacing_meta.json`

## Output Files (Combined Mode)

At stage root:
1. `importance_ranking_q.json`
2. `importance_ranking_j.json`
3. `importance_ranking_combined.json`
4. `importance_ranking.json`
- compatibility alias to combined ranking
5. `importance_heatmap.png`
- from combined ranking
6. `shap_group_heatmap_topk.png`
- from selected sample-level SHAP source (`rollout_j` preferred, fallback `critic_q`)
7. `run_meta.json`
- includes `importance_mode`, weights, ranking source, and sample heatmap source

Per-label outputs remain in:
1. `q_label/...`
2. `j_label/...`

## CLI (Recommended)

```powershell
.\.venv\Scripts\python.exe shap_sensor_find.py `
  --model-dir .\Model_64sensors `
  --buffer-dir .\Buffer_64sensors `
  --episode 110 `
  --importance-mode combined `
  --weight-q 0.5 `
  --weight-j 0.5 `
  --label-mode both `
  --group-size 1 `
  --top-k-select 8 `
  --actuator-exclude-window 2 `
  --outdir .\SHAP_ep110_combined
```

Notes:
1. In combined mode, if `--label-mode` is not `both`, the script auto-upgrades internally to compute both labels.
2. Existing single-mode behavior is still available with `--importance-mode single`.

## Execution Logging and Position+Forcing Visualization

Research runs can be logged in time order using:
1. `run_research_pipeline.py`

Each run writes:
1. `results/<run_id>/Steps_taken.md`
2. `results/<run_id>/...` artifacts (audit JSON, layouts, plots)

Each step entry contains:
1. stage name and step index
2. UTC start/end timestamps
3. exact command (for command steps)
4. exit code and duration
5. key output paths

Position plotting now supports actuator forcing overlay on the same figure (secondary y-axis):

```powershell
.\.venv\Scripts\python.exe plot_sensor_actuator_positions.py `
  --sensor-indices 4,12,20,28,36,44,52,60 `
  --x-file x.dat `
  --a-dim 4 `
  --domain-length 22 `
  --overlay-forcing `
  --output-plot .\RESULTS\positions_bucci8.png `
  --output-json .\RESULTS\positions_bucci8.json `
  --no-show
```
