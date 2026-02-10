# shap_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib
import numpy as np

from KS import KS

matplotlib.use("Agg")


def maybe_group_features_mean(X: np.ndarray, group_size: int) -> Tuple[np.ndarray, List[str]]:
    """
    Optional grouping for correlated spatial sensors.
    If group_size=8 for 64 sensors -> 8 grouped features, each is mean of 8 consecutive sensors.
    """
    d = X.shape[1]
    if group_size <= 1:
        return X, [f"s{i}" for i in range(d)]

    if d % group_size != 0:
        raise ValueError(f"state_dim={d} not divisible by group_size={group_size}")

    g = d // group_size
    Xg = X.reshape(X.shape[0], g, group_size).mean(axis=2)
    names = [f"g{k}_s[{k*group_size}:{(k+1)*group_size-1}]" for k in range(g)]
    return Xg, names


def compute_global_importance(shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """
    I_i = E[ |phi_i| ]
    """
    vals = np.mean(np.abs(shap_values), axis=0)
    imp = {feature_names[i]: float(vals[i]) for i in range(len(feature_names))}
    return dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))


def normalize_importance(importance: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize non-negative importance values so they sum to 1.
    """
    if not importance:
        raise ValueError("importance is empty; cannot normalize.")

    vals: Dict[str, float] = {}
    total = 0.0
    for k, v in importance.items():
        fv = float(v)
        if not np.isfinite(fv):
            raise ValueError(f"importance contains non-finite value for '{k}': {v}")
        if fv < 0.0:
            raise ValueError(f"importance contains negative value for '{k}': {v}")
        vals[k] = fv
        total += fv

    if total <= 0.0:
        raise ValueError("importance sum must be > 0 for normalization.")

    norm = {k: vals[k] / total for k in vals}
    return dict(sorted(norm.items(), key=lambda kv: kv[1], reverse=True))


def combine_importance_weighted(
    importance_q: Dict[str, float],
    importance_j: Dict[str, float],
    w_q: float,
    w_j: float,
) -> Dict[str, float]:
    """
    Weighted combination of normalized Q-importance and J-importance.
    combined_i = w_q * q_norm_i + w_j * j_norm_i
    """
    wq = float(w_q)
    wj = float(w_j)
    if not np.isfinite(wq) or not np.isfinite(wj):
        raise ValueError(f"weights must be finite; got w_q={w_q}, w_j={w_j}")
    if wq < 0.0 or wj < 0.0:
        raise ValueError(f"weights must be non-negative; got w_q={w_q}, w_j={w_j}")
    if wq == 0.0 and wj == 0.0:
        raise ValueError("weights cannot both be zero.")

    keys_q = set(importance_q.keys())
    keys_j = set(importance_j.keys())
    if keys_q != keys_j:
        only_q = sorted(list(keys_q - keys_j))[:5]
        only_j = sorted(list(keys_j - keys_q))[:5]
        raise ValueError(
            "importance key mismatch between Q and J. "
            f"only_in_q(sample)={only_q}, only_in_j(sample)={only_j}"
        )

    q_norm = normalize_importance(importance_q)
    j_norm = normalize_importance(importance_j)

    combined = {
        k: (wq * q_norm[k]) + (wj * j_norm[k])
        for k in q_norm.keys()
    }
    return dict(sorted(combined.items(), key=lambda kv: kv[1], reverse=True))


def plot_group_importance_heatmap(importance: Dict[str, float], outpath: Path):
    import matplotlib.pyplot as plt

    names = list(importance.keys())[:10]
    vals = np.asarray([importance[n] for n in names], dtype=np.float32)
    heat = vals.reshape(1, -1)

    plt.figure(figsize=(max(8, len(names) * 0.9), 2.8))
    im = plt.imshow(heat, cmap="viridis", aspect="auto")
    plt.xticks(np.arange(len(names)), names, rotation=45, ha="right")
    plt.yticks([0], ["mean(|SHAP|)"])
    plt.xlabel("Sensor Group")
    plt.title("Global Group Importance Heatmap")

    for j, v in enumerate(vals):
        plt.text(j, 0, f"{v:.3g}", ha="center", va="center", color="w", fontsize=8)

    plt.colorbar(im, fraction=0.03, pad=0.04, label="mean(|SHAP|)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_sample_group_shap_heatmap(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int,
    outpath: Path,
):
    import matplotlib.pyplot as plt

    shap_values = np.asarray(shap_values)
    if shap_values.ndim != 2:
        raise ValueError(
            f"Expected 2D SHAP array (n_samples, n_features), got shape {shap_values.shape}"
        )
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "Feature name count does not match SHAP feature dimension: "
            f"{len(feature_names)} vs {shap_values.shape[1]}"
        )

    n_samples = shap_values.shape[0]
    k = min(max(1, int(top_k)), n_samples)

    row_score = np.sum(np.abs(shap_values), axis=1)
    row_order = np.argsort(row_score)[::-1][:k]

    col_score = np.mean(np.abs(shap_values), axis=0)
    col_order = np.argsort(col_score)[::-1][:10]

    mat = shap_values[row_order][:, col_order]
    ordered_names = [feature_names[i] for i in col_order]

    vlim = float(np.max(np.abs(mat)))
    if vlim <= 0:
        vlim = 1.0

    plt.figure(figsize=(max(8, len(ordered_names) * 0.9), max(4, k * 0.03)))
    im = plt.imshow(mat, cmap="coolwarm", aspect="auto", vmin=-vlim, vmax=vlim)
    plt.xticks(np.arange(len(ordered_names)), ordered_names, rotation=45, ha="right")
    plt.yticks(np.arange(k), [str(i + 1) for i in range(k)])
    plt.xlabel("Sensor Group (ordered by global mean(|SHAP|))")
    plt.ylabel("Top-K Sample Rank by sum(|SHAP|)")
    plt.title(f"Signed SHAP Heatmap (Top {k} Samples x {len(ordered_names)} Groups)")
    plt.colorbar(im, fraction=0.03, pad=0.04, label="SHAP value (signed)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def select_topk_with_spacing(
    importance: Dict[str, float],
    top_k: int,
    spacing_window: int,
    state_dim: int,
    periodic: bool = True,
) -> List[str]:
    """
    Select top-k sensor features with a spacing constraint.

    Expects feature names like s0, s1, ..., s{state_dim-1}.
    Returns selected feature names in selection order.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if spacing_window < 0:
        raise ValueError("spacing_window must be >= 0")
    if state_dim <= 0:
        raise ValueError("state_dim must be positive")

    ranked = list(importance.items())
    selected: List[str] = []
    selected_idx: List[int] = []

    for name, _ in ranked:
        if not name.startswith("s"):
            raise ValueError(
                f"Spacing selector only supports sensor keys like s<int>, got '{name}'."
            )
        try:
            idx = int(name[1:])
        except ValueError as exc:
            raise ValueError(
                f"Spacing selector only supports sensor keys like s<int>, got '{name}'."
            ) from exc
        if idx < 0 or idx >= state_dim:
            raise ValueError(f"Sensor index out of bounds for state_dim={state_dim}: {idx}")

        keep = True
        for j in selected_idx:
            if periodic:
                d = abs(idx - j)
                d = min(d, state_dim - d)
            else:
                d = abs(idx - j)
            if d <= spacing_window:
                keep = False
                break
        if not keep:
            continue

        selected.append(name)
        selected_idx.append(idx)
        if len(selected) == top_k:
            return selected

    raise ValueError(
        f"Could only select {len(selected)} sensors with top_k={top_k}, "
        f"spacing_window={spacing_window}, periodic={periodic}. "
        "Try reducing top_k or spacing_window."
    )


def actuator_peak_indices(domain_length: float, state_dim: int, action_dim: int) -> List[int]:
    """
    Get actuator center indices from KS forcing basis peaks.
    """
    ks = KS(L=domain_length, N=state_dim, a_dim=action_dim)
    peaks = [int(np.argmax(ks.B[:, i])) for i in range(action_dim)]
    # Keep order but remove duplicates if any.
    out: List[int] = []
    seen = set()
    for p in peaks:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def build_actuator_exclusion_indices(
    actuator_indices: List[int],
    state_dim: int,
    window: int,
    periodic: bool = True,
) -> Set[int]:
    """
    Build excluded sensor indices within +/-window of each actuator index.
    """
    if state_dim <= 0:
        raise ValueError("state_dim must be positive")
    if window < 0:
        raise ValueError("window must be >= 0")
    out: Set[int] = set()
    for a in actuator_indices:
        if a < 0 or a >= state_dim:
            raise ValueError(f"actuator index out of bounds for state_dim={state_dim}: {a}")
        for d in range(-window, window + 1):
            idx = a + d
            if periodic:
                idx %= state_dim
            elif idx < 0 or idx >= state_dim:
                continue
            out.add(int(idx))
    return out


def select_topk_excluding_indices(
    importance: Dict[str, float],
    top_k: int,
    excluded_indices: Set[int],
    state_dim: int,
) -> List[str]:
    """
    Select top-k sensor features after removing excluded indices.

    Expects feature names like s0, s1, ..., s{state_dim-1}.
    Returns selected feature names in ranking order.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if state_dim <= 0:
        raise ValueError("state_dim must be positive")

    selected: List[str] = []
    for name, _ in importance.items():
        if not name.startswith("s"):
            raise ValueError(
                f"Actuator-exclusion selector only supports sensor keys like s<int>, got '{name}'."
            )
        try:
            idx = int(name[1:])
        except ValueError as exc:
            raise ValueError(
                f"Actuator-exclusion selector only supports sensor keys like s<int>, got '{name}'."
            ) from exc
        if idx < 0 or idx >= state_dim:
            raise ValueError(f"Sensor index out of bounds for state_dim={state_dim}: {idx}")
        if idx in excluded_indices:
            continue
        selected.append(name)
        if len(selected) == top_k:
            return selected

    raise ValueError(
        f"Could only select {len(selected)} sensors with top_k={top_k} after excluding "
        f"{len(excluded_indices)} indices near actuators. Try reducing top_k or exclusion window."
    )
