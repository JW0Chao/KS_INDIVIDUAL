# shap_sensor_find.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from KS import KS
from buffer import MemoryBuffer
from train import Trainer
from shap_surrogate import train_surrogate_and_shap
from shap_utils import (
    maybe_group_features_mean,
    compute_global_importance,
    combine_importance_weighted,
    plot_group_importance_heatmap,
    plot_sample_group_shap_heatmap,
    actuator_peak_indices,
    build_actuator_exclusion_indices,
    select_topk_excluding_indices,
)


@torch.no_grad()
def compute_q_of_pi(
    trainer: Trainer,
    states_np: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Compute y = Q(s, pi(s)) for a batch of states.
    states_np: (N, state_dim)
    returns: (N,)
    """
    trainer.actor.eval()
    trainer.critic.eval()

    ys: List[np.ndarray] = []
    n = states_np.shape[0]
    for i in range(0, n, batch_size):
        s = torch.as_tensor(states_np[i:i + batch_size], dtype=torch.float32, device=device)
        a = trainer.actor(s)
        q = trainer.critic(s, a)
        ys.append(q.view(-1).detach().cpu().numpy())
    return np.concatenate(ys, axis=0).astype(np.float32)


@torch.no_grad()
def compute_rollout_j(
    trainer: Trainer,
    states_np: np.ndarray,
    device: torch.device,
    action_dim: int,
    horizon: int,
    target_state: np.ndarray,
    domain_length: float,
    metric: str,
    gamma: float,
) -> np.ndarray:
    """
    Compute rollout-based label J for each state by simulating closed-loop KS for H steps.
    Supported metrics:
      - mean_l2
      - discounted_l2
      - terminal_l2
    """
    if horizon <= 0:
        raise ValueError("rollout horizon must be positive")
    if metric == "discounted_l2" and not (0.0 < gamma <= 1.0):
        raise ValueError(f"j_gamma must be in (0,1] for discounted_l2, got {gamma}")

    trainer.actor.eval()
    labels = np.zeros((states_np.shape[0],), dtype=np.float32)

    for i in range(states_np.shape[0]):
        ks = KS(L=domain_length, N=states_np.shape[1], a_dim=action_dim)
        obs = np.float32(states_np[i].copy())

        if metric == "mean_l2":
            accum = 0.0
        elif metric == "discounted_l2":
            accum = 0.0
            disc = 1.0
        else:  # terminal_l2
            accum = 0.0

        last_dev = 0.0
        for _ in range(horizon):
            s = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = trainer.actor(s).squeeze(0).cpu().numpy()
            obs = ks.advance(obs, action)
            dev = float(np.linalg.norm(obs - target_state))
            last_dev = dev

            if metric == "mean_l2":
                accum += dev
            elif metric == "discounted_l2":
                accum += disc * dev
                disc *= gamma

        if metric == "mean_l2":
            labels[i] = np.float32(accum / horizon)
        elif metric == "discounted_l2":
            labels[i] = np.float32(accum)
        else:  # terminal_l2
            labels[i] = np.float32(last_dev)

    return labels


def sample_states_from_buffer(ram: MemoryBuffer, n_states: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks: List[np.ndarray] = []
    remaining = n_states
    while remaining > 0:
        bs = min(4096, remaining)
        s, _, _, _ = ram.sample(bs)
        chunks.append(np.asarray(s))
        remaining -= bs
    x = np.concatenate(chunks, axis=0)
    x = x[rng.permutation(x.shape[0])]
    return x


def _parse_index_list(expr: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for raw in expr.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if "-" in tok:
            a_str, b_str = tok.split("-", 1)
            a = int(a_str.strip())
            b = int(b_str.strip())
            if b < a:
                raise ValueError(f"Invalid range '{tok}': end < start")
            for i in range(a, b + 1):
                if i not in seen:
                    seen.add(i)
                    out.append(i)
        else:
            i = int(tok)
            if i not in seen:
                seen.add(i)
                out.append(i)
    if not out:
        raise ValueError("No valid indices parsed.")
    return out


def _groups_to_sensor_indices(groups_expr: str, group_size: int, state_dim: int) -> List[int]:
    groups: List[int] = []
    seen = set()
    for raw in groups_expr.split(","):
        tok = raw.strip().lower()
        if not tok:
            continue
        if tok.startswith("g"):
            tok = tok[1:]
        gid = int(tok)
        if gid < 0:
            raise ValueError(f"Invalid group id '{raw}'")
        if gid not in seen:
            seen.add(gid)
            groups.append(gid)
    if not groups:
        raise ValueError("No valid groups parsed.")

    max_group = (state_dim // group_size) - 1
    out: List[int] = []
    for gid in groups:
        if gid > max_group:
            raise ValueError(
                f"Group g{gid} out of bounds for state_dim={state_dim}, "
                f"group_size={group_size} (max group g{max_group})"
            )
        start = gid * group_size
        out.extend(list(range(start, start + group_size)))
    return out


def _base_meta(
    args: argparse.Namespace,
    label_mode_name: str,
    base_value_json: Any,
    device: torch.device,
    selected_sensor_indices: Optional[List[int]],
    selected_sensor_groups: Optional[List[str]],
    topk_raw: Optional[List[int]],
    topk_spacing: Optional[List[int]],
    ranking_source: str,
    sample_heatmap_source_label: str,
    stage_name: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "episode": args.episode,
        "model_dir": args.model_dir,
        "buffer_dir": args.buffer_dir,
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "n_train": args.n_train,
        "n_explain": args.n_explain,
        "group_size": args.group_size,
        "surrogate": args.surrogate,
        "base_value": base_value_json,
        "device": str(device),
        "plot_mode": "heatmap_only",
        "sample_heatmap_top_k": args.sample_heatmap_top_k,
        "sample_heatmap_signed": True,
        "selected_sensor_indices": selected_sensor_indices,
        "selected_sensor_groups": selected_sensor_groups,
        "label_mode": label_mode_name,
        "requested_label_mode": getattr(args, "requested_label_mode", args.label_mode),
        "effective_label_mode": getattr(args, "effective_label_mode", args.label_mode),
        "importance_mode": args.importance_mode,
        "weight_q": args.weight_q,
        "weight_j": args.weight_j,
        "ranking_source": ranking_source,
        "sample_heatmap_source_label": sample_heatmap_source_label,
        "rollout_horizon": args.rollout_horizon,
        "j_metric": args.j_metric,
        "j_gamma": args.j_gamma,
        "top_k_select": args.top_k_select,
        "spacing_window": args.actuator_exclude_window,
        "spacing_periodic": args.actuator_exclude_periodic,
        "actuator_indices": getattr(args, "actuator_indices", None),
        "actuator_exclusion_window": args.actuator_exclude_window,
        "actuator_exclusion_periodic": args.actuator_exclude_periodic,
        "topk_raw": topk_raw,
        "topk_spacing": topk_spacing,
        "topk_actuator_excluded": topk_spacing,
        "auto_two_stage": args.auto_two_stage,
        "coarse_group_size": args.coarse_group_size,
        "coarse_top_groups": args.coarse_top_groups,
        "stage_name": stage_name,
    }


def _write_topk_files(
    outdir: Path,
    importance: Dict[str, float],
    args: argparse.Namespace,
    ranking_source: str,
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    topk_raw: Optional[List[int]] = None
    topk_spacing: Optional[List[int]] = None

    if args.group_size != 1:
        print("[WARN] group_size != 1, skipping topk_raw/topk_actuator_excluded sensor files.")
        return topk_raw, topk_spacing

    raw_names = list(importance.keys())[:args.top_k_select]
    topk_raw = [int(n[1:]) for n in raw_names]
    (outdir / "topk_raw.json").write_text(json.dumps(topk_raw, indent=2), encoding="utf-8")

    actuator_indices = actuator_peak_indices(
        domain_length=args.domain_length,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    )
    args.actuator_indices = actuator_indices

    spacing_names = None
    effective_spacing_window = args.actuator_exclude_window
    last_spacing_error: Optional[Exception] = None
    for window in range(args.actuator_exclude_window, -1, -1):
        try:
            excluded = build_actuator_exclusion_indices(
                actuator_indices=actuator_indices,
                state_dim=args.state_dim,
                window=window,
                periodic=args.actuator_exclude_periodic,
            )
            spacing_names = select_topk_excluding_indices(
                importance=importance,
                top_k=args.top_k_select,
                excluded_indices=excluded,
                state_dim=args.state_dim,
            )
            effective_spacing_window = window
            break
        except ValueError as exc:
            last_spacing_error = exc
            continue
    if spacing_names is None:
        assert last_spacing_error is not None
        raise last_spacing_error
    if effective_spacing_window != args.actuator_exclude_window:
        print(
            "[WARN] Requested actuator exclusion window="
            f"{args.actuator_exclude_window} could not select top_k={args.top_k_select}; "
            f"used actuator exclusion window={effective_spacing_window} instead."
        )
    topk_spacing = [int(n[1:]) for n in spacing_names]
    (outdir / "topk_actuator_excluded.json").write_text(
        json.dumps(topk_spacing, indent=2), encoding="utf-8"
    )
    # Backward-compatible aliases.
    (outdir / "topk_spacing.json").write_text(json.dumps(topk_spacing, indent=2), encoding="utf-8")
    meta_obj = {
        "requested_top_k": args.top_k_select,
        "requested_actuator_exclusion_window": args.actuator_exclude_window,
        "effective_actuator_exclusion_window": effective_spacing_window,
        "actuator_exclusion_periodic": args.actuator_exclude_periodic,
        "actuator_indices": actuator_indices,
        "ranking_source": ranking_source,
        "importance_mode": args.importance_mode,
        "weight_q": args.weight_q,
        "weight_j": args.weight_j,
        "deprecated_alias_files": ["topk_spacing.json", "topk_spacing_meta.json"],
    }
    (outdir / "topk_actuator_excluded_meta.json").write_text(
        json.dumps(meta_obj, indent=2),
        encoding="utf-8",
    )
    (outdir / "topk_spacing_meta.json").write_text(
        json.dumps(
            meta_obj,
            indent=2,
        ),
        encoding="utf-8",
    )
    return topk_raw, topk_spacing


def _run_single_label(
    outdir: Path,
    x_train_g: np.ndarray,
    y_train: np.ndarray,
    x_explain_g: np.ndarray,
    feature_names: List[str],
    args: argparse.Namespace,
    device: torch.device,
    selected_sensor_indices: Optional[List[int]],
    selected_sensor_groups: Optional[List[str]],
    label_mode_name: str,
    stage_name: Optional[str] = None,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)

    results = train_surrogate_and_shap(
        X_train=x_train_g,
        y_train=y_train,
        X_explain=x_explain_g,
        surrogate=args.surrogate,
        test_split=args.test_split,
        seed=args.seed,
        feature_names=feature_names,
        outdir=outdir,
        save_beeswarm=False,
    )

    shap_values = np.asarray(results["shap_values"])
    if shap_values.ndim != 2:
        raise ValueError(
            "Unexpected SHAP shape; expected 2D array (n_samples, n_features), "
            f"got {shap_values.shape}"
        )
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "SHAP feature dimension mismatch: "
            f"shap_values has {shap_values.shape[1]} features, "
            f"feature_names has {len(feature_names)}"
        )

    base_value = results["base_value"]
    if np.isscalar(base_value):
        base_value_json: Any = float(base_value)
    else:
        base_arr = np.asarray(base_value)
        base_value_json = float(base_arr.reshape(-1)[0]) if base_arr.size == 1 else base_arr.tolist()

    importance = compute_global_importance(shap_values, feature_names)
    (outdir / "importance_ranking.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")
    (outdir / "surrogate_metrics.json").write_text(json.dumps(results["metrics"], indent=2), encoding="utf-8")

    plot_group_importance_heatmap(importance, outpath=outdir / "importance_heatmap.png")
    plot_sample_group_shap_heatmap(
        shap_values=shap_values,
        feature_names=feature_names,
        top_k=args.sample_heatmap_top_k,
        outpath=outdir / "shap_group_heatmap_topk.png",
    )

    topk_raw, topk_spacing = _write_topk_files(
        outdir=outdir,
        importance=importance,
        args=args,
        ranking_source=label_mode_name,
    )

    meta = _base_meta(
        args=args,
        label_mode_name=label_mode_name,
        base_value_json=base_value_json,
        device=device,
        selected_sensor_indices=selected_sensor_indices,
        selected_sensor_groups=selected_sensor_groups,
        topk_raw=topk_raw,
        topk_spacing=topk_spacing,
        ranking_source=label_mode_name,
        sample_heatmap_source_label=label_mode_name,
        stage_name=stage_name,
    )
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "importance": importance,
        "metrics": results["metrics"],
        "topk_raw": topk_raw,
        "topk_spacing": topk_spacing,
        "shap_values": shap_values,
        "feature_names": feature_names,
        "base_value_json": base_value_json,
    }


def main() -> None:
    ap = argparse.ArgumentParser()

    # checkpoints
    ap.add_argument("--model-dir", type=str, required=True, help="Directory containing *_actor.pt and *_critic.pt")
    ap.add_argument("--buffer-dir", type=str, required=True, help="Directory containing buffer pickle files and reward.dat")
    ap.add_argument("--episode", type=int, required=True, help="Episode index, e.g. 5000_actor.pt, 5000buffer.txt.")

    # dims
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--action-dim", type=int, default=4)
    ap.add_argument("--action-lim", type=float, default=0.5)

    # dataset sizes
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-explain", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)

    # grouping / focus
    ap.add_argument("--group-size", type=int, default=1, help="If >1, average consecutive sensors into blocks.")
    ap.add_argument("--auto-two-stage", dest="auto_two_stage", action="store_true", default=True)
    ap.add_argument("--no-auto-two-stage", dest="auto_two_stage", action="store_false")
    ap.add_argument("--coarse-group-size", type=int, default=8)
    ap.add_argument("--coarse-top-groups", type=int, default=3)
    ap.add_argument("--sensor-indices", type=str, default=None, help="Optional subset, e.g. '0-7,24-31,40-47'.")
    ap.add_argument("--sensor-groups", type=str, default=None, help="Optional groups, e.g. 'g3,g0,g5'.")
    ap.add_argument("--sensor-group-size", type=int, default=8, help="Group size used to expand --sensor-groups.")

    # surrogate choice
    ap.add_argument("--surrogate", type=str, default="xgb", choices=["xgb", "lgbm", "rf"])
    ap.add_argument("--test-split", type=float, default=0.2)

    # labels
    ap.add_argument("--label-mode", type=str, default="critic_q", choices=["critic_q", "rollout_j", "both"])
    ap.add_argument("--importance-mode", type=str, default="combined", choices=["single", "combined"])
    ap.add_argument("--weight-q", type=float, default=0.5)
    ap.add_argument("--weight-j", type=float, default=0.5)
    ap.add_argument("--rollout-horizon", type=int, default=300)
    ap.add_argument("--target-file", type=str, default="u3.dat")
    ap.add_argument("--x-file", type=str, default="x.dat")
    ap.add_argument("--domain-length", type=float, default=22.0)
    ap.add_argument("--j-metric", type=str, default="mean_l2", choices=["mean_l2", "discounted_l2", "terminal_l2"])
    ap.add_argument("--j-gamma", type=float, default=0.99)

    # top-k select
    ap.add_argument("--top-k-select", type=int, default=8)
    ap.add_argument("--actuator-exclude-window", type=int, default=2)
    ap.add_argument("--actuator-exclude-periodic", dest="actuator_exclude_periodic", action="store_true", default=True)
    ap.add_argument("--no-actuator-exclude-periodic", dest="actuator_exclude_periodic", action="store_false")
    ap.add_argument(
        "--spacing-window",
        type=int,
        default=None,
        help="Deprecated alias for --actuator-exclude-window.",
    )
    ap.add_argument(
        "--spacing-periodic",
        dest="spacing_periodic",
        action="store_true",
        default=None,
        help="Deprecated alias for --actuator-exclude-periodic.",
    )
    ap.add_argument(
        "--no-spacing-periodic",
        dest="spacing_periodic",
        action="store_false",
        help="Deprecated alias for --no-actuator-exclude-periodic.",
    )

    # plotting
    ap.add_argument("--sample-heatmap-top-k", type=int, default=200)
    ap.add_argument("--outdir", type=str, default=None)

    args = ap.parse_args()
    if args.spacing_window is not None:
        print(
            "[WARN] --spacing-window is deprecated; using it as "
            "--actuator-exclude-window."
        )
        args.actuator_exclude_window = args.spacing_window
    if args.spacing_periodic is not None:
        print(
            "[WARN] --spacing-periodic/--no-spacing-periodic are deprecated; using them as "
            "--actuator-exclude-periodic/--no-actuator-exclude-periodic."
        )
        args.actuator_exclude_periodic = args.spacing_periodic
    if args.actuator_exclude_window < 0:
        raise ValueError(f"actuator-exclude-window must be >= 0, got {args.actuator_exclude_window}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not np.isfinite(args.weight_q) or not np.isfinite(args.weight_j):
        raise ValueError(f"Weights must be finite; got weight_q={args.weight_q}, weight_j={args.weight_j}")
    if args.weight_q < 0.0 or args.weight_j < 0.0:
        raise ValueError(f"Weights must be non-negative; got weight_q={args.weight_q}, weight_j={args.weight_j}")
    if args.importance_mode == "combined" and args.weight_q == 0.0 and args.weight_j == 0.0:
        raise ValueError("weight_q and weight_j cannot both be zero in combined mode.")

    model_dir = Path(args.model_dir)
    buffer_dir = Path(args.buffer_dir)
    outdir = Path(args.outdir) if args.outdir else Path(f"./SHAP_ep{args.episode}")
    outdir.mkdir(parents=True, exist_ok=True)

    requested_label_mode = args.label_mode
    effective_label_mode = args.label_mode
    if args.importance_mode == "combined" and requested_label_mode != "both":
        print(
            "[INFO] importance-mode=combined requires both labels; "
            f"auto-upgrading label-mode from '{requested_label_mode}' to 'both'."
        )
        effective_label_mode = "both"
    args.requested_label_mode = requested_label_mode
    args.effective_label_mode = effective_label_mode

    if args.sensor_indices and args.sensor_groups:
        raise ValueError("Use only one of --sensor-indices or --sensor-groups.")
    if args.auto_two_stage and (args.sensor_indices or args.sensor_groups):
        raise ValueError(
            "--sensor-indices/--sensor-groups are for single-stage runs. "
            "Use --no-auto-two-stage to run manually with those options."
        )
    if args.auto_two_stage and args.coarse_group_size <= 1:
        raise ValueError("--coarse-group-size must be > 1 in auto two-stage mode.")
    if args.auto_two_stage and args.state_dim % args.coarse_group_size != 0:
        raise ValueError(
            f"state_dim={args.state_dim} not divisible by coarse_group_size={args.coarse_group_size}"
        )

    ram = MemoryBuffer(buffer_dir=str(buffer_dir))
    ram.load_buffer(args.episode)

    trainer = Trainer(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        action_lim=args.action_lim,
        ram=ram,
        device=device,
        Test=True,
        model_dir=str(model_dir),
        buffer_dir=str(buffer_dir),
    )
    trainer.load_models(args.episode, Test=True)

    x_train = sample_states_from_buffer(ram, args.n_train, seed=args.seed)
    x_explain = sample_states_from_buffer(ram, args.n_explain, seed=args.seed + 1)

    manual_selected_sensor_indices = None
    manual_selected_sensor_groups = None
    if args.sensor_indices:
        manual_selected_sensor_indices = _parse_index_list(args.sensor_indices)
    elif args.sensor_groups:
        manual_selected_sensor_indices = _groups_to_sensor_indices(
            groups_expr=args.sensor_groups,
            group_size=args.sensor_group_size,
            state_dim=args.state_dim,
        )
        manual_selected_sensor_groups = [g.strip() for g in args.sensor_groups.split(",") if g.strip()]

    if effective_label_mode in ("rollout_j", "both"):
        x_grid = np.loadtxt(args.x_file)
        if x_grid.ndim != 1 or x_grid.shape[0] != args.state_dim:
            raise ValueError(
                f"x-file shape mismatch: expected 1D length {args.state_dim}, got {x_grid.shape}"
            )
        u_target = np.loadtxt(args.target_file)
        if u_target.ndim != 1 or u_target.shape[0] != args.state_dim:
            raise ValueError(
                f"target-file shape mismatch: expected 1D length {args.state_dim}, got {u_target.shape}"
            )
    else:
        u_target = None

    labels: Dict[str, np.ndarray] = {}
    if effective_label_mode in ("critic_q", "both"):
        labels["critic_q"] = compute_q_of_pi(trainer, x_train, device=device)
    if effective_label_mode in ("rollout_j", "both"):
        assert u_target is not None
        labels["rollout_j"] = compute_rollout_j(
            trainer=trainer,
            states_np=x_train,
            device=device,
            action_dim=args.action_dim,
            horizon=args.rollout_horizon,
            target_state=u_target,
            domain_length=args.domain_length,
            metric=args.j_metric,
            gamma=args.j_gamma,
        )

    for label_name, y_train in labels.items():
        if not np.all(np.isfinite(y_train)):
            raise ValueError(f"Non-finite labels detected for {label_name}.")

    def build_feature_view(
        stage_group_size: int,
        stage_selected_sensor_indices: Optional[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if stage_selected_sensor_indices is not None:
            if stage_group_size != 1:
                raise ValueError("When selecting explicit sensors, stage group size must be 1.")
            bad = [i for i in stage_selected_sensor_indices if i < 0 or i >= args.state_dim]
            if bad:
                raise ValueError(f"Sensor indices out of bounds for state_dim={args.state_dim}: {bad}")
            x_train_stage = x_train[:, stage_selected_sensor_indices]
            x_explain_stage = x_explain[:, stage_selected_sensor_indices]
            names = [f"s{i}" for i in stage_selected_sensor_indices]
            return x_train_stage, x_explain_stage, names
        x_train_stage, names = maybe_group_features_mean(x_train, group_size=stage_group_size)
        x_explain_stage, _ = maybe_group_features_mean(x_explain, group_size=stage_group_size)
        return x_train_stage, x_explain_stage, names

    def run_stage(
        stage_outdir: Path,
        stage_name: str,
        stage_group_size: int,
        stage_selected_sensor_indices: Optional[List[int]],
        stage_selected_sensor_groups: Optional[List[str]],
    ) -> Dict[str, Any]:
        stage_outdir.mkdir(parents=True, exist_ok=True)
        stage_args = argparse.Namespace(**vars(args))
        stage_args.group_size = stage_group_size
        stage_args.label_mode = effective_label_mode

        x_train_stage, x_explain_stage, feature_names_stage = build_feature_view(
            stage_group_size=stage_group_size,
            stage_selected_sensor_indices=stage_selected_sensor_indices,
        )

        stage_outputs: Dict[str, Dict[str, Any]] = {}
        for label_name, y_train in labels.items():
            if effective_label_mode == "both":
                label_outdir = stage_outdir / ("q_label" if label_name == "critic_q" else "j_label")
            else:
                label_outdir = stage_outdir
            stage_outputs[label_name] = _run_single_label(
                outdir=label_outdir,
                x_train_g=x_train_stage,
                y_train=y_train,
                x_explain_g=x_explain_stage,
                feature_names=feature_names_stage,
                args=stage_args,
                device=device,
                selected_sensor_indices=stage_selected_sensor_indices,
                selected_sensor_groups=stage_selected_sensor_groups,
                label_mode_name=label_name,
                stage_name=stage_name,
            )
        if args.importance_mode == "combined":
            if "critic_q" not in stage_outputs or "rollout_j" not in stage_outputs:
                raise ValueError(
                    "importance-mode=combined requires both critic_q and rollout_j outputs."
                )
            importance_q = stage_outputs["critic_q"]["importance"]
            importance_j = stage_outputs["rollout_j"]["importance"]
            importance_combined = combine_importance_weighted(
                importance_q=importance_q,
                importance_j=importance_j,
                w_q=args.weight_q,
                w_j=args.weight_j,
            )

            (stage_outdir / "importance_ranking_q.json").write_text(
                json.dumps(importance_q, indent=2), encoding="utf-8"
            )
            (stage_outdir / "importance_ranking_j.json").write_text(
                json.dumps(importance_j, indent=2), encoding="utf-8"
            )
            (stage_outdir / "importance_ranking_combined.json").write_text(
                json.dumps(importance_combined, indent=2), encoding="utf-8"
            )
            (stage_outdir / "importance_ranking.json").write_text(
                json.dumps(importance_combined, indent=2), encoding="utf-8"
            )

            sample_label = "rollout_j" if "rollout_j" in stage_outputs else "critic_q"
            sample_out = stage_outputs[sample_label]
            plot_group_importance_heatmap(importance_combined, outpath=stage_outdir / "importance_heatmap.png")
            plot_sample_group_shap_heatmap(
                shap_values=np.asarray(sample_out["shap_values"]),
                feature_names=sample_out["feature_names"],
                top_k=args.sample_heatmap_top_k,
                outpath=stage_outdir / "shap_group_heatmap_topk.png",
            )

            stage_topk_raw, stage_topk_spacing = _write_topk_files(
                outdir=stage_outdir,
                importance=importance_combined,
                args=stage_args,
                ranking_source="combined",
            )

            stage_meta = _base_meta(
                args=stage_args,
                label_mode_name="combined",
                base_value_json=sample_out["base_value_json"],
                device=device,
                selected_sensor_indices=stage_selected_sensor_indices,
                selected_sensor_groups=stage_selected_sensor_groups,
                topk_raw=stage_topk_raw,
                topk_spacing=stage_topk_spacing,
                ranking_source="combined",
                sample_heatmap_source_label=sample_label,
                stage_name=stage_name,
            )
            (stage_outdir / "run_meta.json").write_text(
                json.dumps(stage_meta, indent=2), encoding="utf-8"
            )
            return {
                "label_outputs": stage_outputs,
                "selection_importance": importance_combined,
                "selection_source": "combined",
                "topk_raw": stage_topk_raw,
                "topk_spacing": stage_topk_spacing,
                "sample_heatmap_source_label": sample_label,
            }

        source_label = "rollout_j" if "rollout_j" in stage_outputs else "critic_q"
        return {
            "label_outputs": stage_outputs,
            "selection_importance": stage_outputs[source_label]["importance"],
            "selection_source": source_label,
            "topk_raw": stage_outputs[source_label]["topk_raw"],
            "topk_spacing": stage_outputs[source_label]["topk_spacing"],
            "sample_heatmap_source_label": source_label,
        }

    if args.auto_two_stage:
        coarse_stage_dir = outdir / f"coarse_g{args.coarse_group_size}"
        coarse_stage = run_stage(
            stage_outdir=coarse_stage_dir,
            stage_name=f"coarse_g{args.coarse_group_size}",
            stage_group_size=args.coarse_group_size,
            stage_selected_sensor_indices=None,
            stage_selected_sensor_groups=None,
        )
        coarse_importance = coarse_stage["selection_importance"]
        source_label = coarse_stage["selection_source"]
        top_group_names = list(coarse_importance.keys())[:args.coarse_top_groups]
        if not top_group_names:
            raise ValueError("No coarse groups found to build fine stage.")

        group_ids: List[int] = []
        for name in top_group_names:
            if not name.startswith("g"):
                raise ValueError(
                    f"Expected grouped feature name starting with 'g', got '{name}'. "
                    "Coarse stage must be grouped."
                )
            gid = int(name[1:].split("_", 1)[0])
            group_ids.append(gid)

        fine_sensor_indices: List[int] = []
        for gid in group_ids:
            start = gid * args.coarse_group_size
            fine_sensor_indices.extend(list(range(start, start + args.coarse_group_size)))
        fine_sensor_indices = sorted(set(fine_sensor_indices))
        fine_sensor_groups = [f"g{g}" for g in group_ids]

        fine_stage_dir = outdir / f"fine_g1_from_g{args.coarse_group_size}"
        fine_stage = run_stage(
            stage_outdir=fine_stage_dir,
            stage_name=f"fine_g1_from_g{args.coarse_group_size}",
            stage_group_size=1,
            stage_selected_sensor_indices=fine_sensor_indices,
            stage_selected_sensor_groups=fine_sensor_groups,
        )

        pipeline_summary = {
            "mode": "auto_two_stage",
            "coarse_group_size": args.coarse_group_size,
            "coarse_top_groups": args.coarse_top_groups,
            "coarse_source_label": source_label,
            "importance_mode": args.importance_mode,
            "weight_q": args.weight_q,
            "weight_j": args.weight_j,
            "coarse_top_group_names": top_group_names,
            "fine_sensor_indices": fine_sensor_indices,
            "fine_sensor_groups": fine_sensor_groups,
            "labels_run": list(labels.keys()),
            "fine_selection_source": fine_stage["selection_source"],
            "coarse_stage_dir": str(coarse_stage_dir),
            "fine_stage_dir": str(fine_stage_dir),
        }
        (outdir / "pipeline_summary.json").write_text(json.dumps(pipeline_summary, indent=2), encoding="utf-8")
    else:
        run_stage(
            stage_outdir=outdir,
            stage_name="single_stage",
            stage_group_size=args.group_size,
            stage_selected_sensor_indices=manual_selected_sensor_indices,
            stage_selected_sensor_groups=manual_selected_sensor_groups,
        )

    print(f"[DONE] SHAP sensor ranking saved to: {outdir}")
    print("Next step: evaluate selected top-k sensors in closed-loop.")


if __name__ == "__main__":
    main()
