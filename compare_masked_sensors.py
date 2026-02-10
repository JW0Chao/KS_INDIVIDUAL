from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from KS import KS
import model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare top-k SHAP sensors vs random-k sensors using masked reconstruction."
    )
    ap.add_argument("--models-dir", type=str, required=True)
    ap.add_argument("--episode", type=int, default=None, help="Checkpoint episode to load. Default: latest.")
    ap.add_argument("--importance-json", type=str, default=None)
    ap.add_argument("--top-sensor-json", type=str, default=None, help="Optional JSON list of selected sensors.")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--a-dim", type=int, default=4)
    ap.add_argument("--a-max", type=float, default=0.5)
    ap.add_argument("--num-evals", type=int, default=20)
    ap.add_argument("--num-random-sets", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--domain-length", type=float, default=22.0)
    ap.add_argument("--target-file", type=str, default="u3.dat")
    ap.add_argument("--x-file", type=str, default="x.dat")
    ap.add_argument("--init-file", type=str, default="INIT.dat")
    ap.add_argument("--stabilize-threshold", type=float, default=5.0)
    ap.add_argument("--stabilize-hold", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--no-show", action="store_true")
    return ap.parse_args()


def pick_checkpoint_episode(models_dir: str, requested_episode: int | None) -> int:
    if requested_episode is not None:
        actor_path = os.path.join(models_dir, f"{requested_episode}_actor.pt")
        critic_path = os.path.join(models_dir, f"{requested_episode}_critic.pt")
        if not (os.path.exists(actor_path) and os.path.exists(critic_path)):
            raise FileNotFoundError(
                f"Missing checkpoint files for episode {requested_episode}: "
                f"{actor_path}, {critic_path}"
            )
        return requested_episode

    actor_ckpts = [
        f for f in os.listdir(models_dir)
        if f.endswith("_actor.pt") and not f.endswith("_target_actor.pt")
    ]
    if not actor_ckpts:
        raise FileNotFoundError(f"No actor checkpoints found in {models_dir}")
    return max(int(f.split("_")[0]) for f in actor_ckpts)


def first_stable_time(deviation_curve: np.ndarray, threshold: float, hold_steps: int) -> float:
    n = deviation_curve.shape[0]
    if hold_steps <= 1:
        hits = np.where(deviation_curve <= threshold)[0]
        return float(hits[0]) if hits.size > 0 else np.nan
    for t in range(0, n - hold_steps + 1):
        if np.all(deviation_curve[t:t + hold_steps] <= threshold):
            return float(t)
    return np.nan


def load_top_sensor_indices(importance_json: str, top_k: int, state_dim: int) -> List[int]:
    p = Path(importance_json)
    if not p.exists():
        raise FileNotFoundError(f"importance json not found: {importance_json}")
    imp = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(imp, dict) or not imp:
        raise ValueError("importance json must be a non-empty object mapping feature->importance")

    pairs: List[Tuple[int, float]] = []
    for k, v in imp.items():
        if not isinstance(k, str) or not k.startswith("s"):
            raise ValueError(f"Malformed feature key '{k}'. Expected keys like s4, s25.")
        try:
            idx = int(k[1:])
        except ValueError as exc:
            raise ValueError(f"Malformed feature key '{k}'. Expected keys like s4, s25.") from exc
        if idx < 0 or idx >= state_dim:
            raise ValueError(f"Feature index out of bounds for state_dim={state_dim}: {idx}")
        pairs.append((idx, float(v)))

    pairs.sort(key=lambda x: x[1], reverse=True)
    if len(pairs) < top_k:
        raise ValueError(
            f"Requested top_k={top_k} but only {len(pairs)} features available in {importance_json}"
        )
    return [idx for idx, _ in pairs[:top_k]]


def load_sensor_indices_json(path: str, state_dim: int) -> List[int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"top sensor json not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "indices" in data:
            data = data["indices"]
        else:
            raise ValueError("top sensor json dict must contain key 'indices'.")
    if not isinstance(data, list):
        raise ValueError("top sensor json must be a JSON list of sensor indices.")

    out: List[int] = []
    seen = set()
    for item in data:
        if isinstance(item, str) and item.startswith("s"):
            idx = int(item[1:])
        else:
            idx = int(item)
        if idx < 0 or idx >= state_dim:
            raise ValueError(f"Sensor index out of bounds for state_dim={state_dim}: {idx}")
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    if not out:
        raise ValueError("top sensor json is empty.")
    return out


def make_random_sensor_sets(
    state_dim: int,
    set_size: int,
    num_sets: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    if set_size > state_dim:
        raise ValueError(f"set_size={set_size} cannot exceed state_dim={state_dim}")

    unique_sets = set()
    out: List[List[int]] = []
    max_unique = int(math.comb(state_dim, set_size))
    target = min(num_sets, max_unique)

    while len(out) < target:
        idx = tuple(sorted(rng.choice(state_dim, size=set_size, replace=False).tolist()))
        if idx in unique_sets:
            continue
        unique_sets.add(idx)
        out.append(list(idx))

    return out


def periodic_linear_reconstruct(
    u_true: np.ndarray,
    x: np.ndarray,
    sensor_idx: np.ndarray,
    domain_length: float,
) -> np.ndarray:
    if sensor_idx.ndim != 1 or sensor_idx.size < 2:
        raise ValueError("Need at least two sensor points for interpolation.")

    sidx = np.unique(sensor_idx.astype(np.int64))
    x_known = x[sidx]
    y_known = u_true[sidx]

    order = np.argsort(x_known)
    x_sorted = x_known[order]
    y_sorted = y_known[order]

    x_ext = np.concatenate(([x_sorted[-1] - domain_length], x_sorted, [x_sorted[0] + domain_length]))
    y_ext = np.concatenate(([y_sorted[-1]], y_sorted, [y_sorted[0]]))

    u_rec = np.interp(x, x_ext, y_ext).astype(np.float32)
    # Enforce exact observed values at selected sensors.
    u_rec[sidx] = u_true[sidx]
    return u_rec


def run_condition(
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    x: np.ndarray,
    u_target: np.ndarray,
    init_states: np.ndarray,
    init_rows: np.ndarray,
    sensor_idx: List[int],
    a_dim: int,
    max_steps: int,
    stabilize_threshold: float,
    stabilize_hold: int,
    domain_length: float,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    if len(sensor_idx) < 2:
        raise ValueError("sensor_idx must contain at least two indices")

    sensor_arr = np.asarray(sensor_idx, dtype=np.int64)
    n_eval = init_rows.shape[0]
    value_curves = np.zeros((n_eval, max_steps), dtype=np.float32)
    deviation_curves = np.zeros((n_eval, max_steps), dtype=np.float32)
    stable_steps = np.full((n_eval,), np.nan, dtype=np.float32)

    ks = KS(L=domain_length, N=x.size, a_dim=a_dim)

    with torch.no_grad():
        for e in range(n_eval):
            obs = np.float32(init_states[int(init_rows[e])].copy())
            for t in range(max_steps):
                state = periodic_linear_reconstruct(obs, x, sensor_arr, domain_length)
                st = torch.from_numpy(state).to(device).unsqueeze(0)
                at = actor(st)
                qt = critic(st, at)
                action = at.squeeze(0).cpu().numpy()

                value_curves[e, t] = float(qt.item())
                obs = ks.advance(obs, action)
                deviation_curves[e, t] = float(np.linalg.norm(obs - u_target))

            stable_steps[e] = first_stable_time(
                deviation_curves[e], threshold=stabilize_threshold, hold_steps=stabilize_hold
            )

    return {
        "value_curves": value_curves,
        "deviation_curves": deviation_curves,
        "stable_steps": stable_steps,
    }


def _mean_curve(a: np.ndarray) -> np.ndarray:
    return np.nanmean(a, axis=0)


def _median_curve(a: np.ndarray) -> np.ndarray:
    return np.nanmedian(a, axis=0)


def plot_comparison(
    time_axis: np.ndarray,
    top_metrics: Dict[str, np.ndarray],
    random_set_metrics: List[Dict[str, np.ndarray]],
    stable_top_time: np.ndarray,
    stable_random_time_by_set: np.ndarray,
    outdir: Path,
    no_show: bool,
) -> None:
    # Value vs time
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    top_mean = _mean_curve(top_metrics["value_curves"])
    top_median = _median_curve(top_metrics["value_curves"])
    ax1.plot(time_axis, top_mean, color="tab:blue", linewidth=2.2, label="Model Top Sensors (mean)")
    ax1.plot(time_axis, top_median, color="tab:blue", linewidth=2.2, linestyle="--", label="Model Top Sensors (median)")

    rand_means = []
    rand_medians = []
    for m in random_set_metrics:
        rm = _mean_curve(m["value_curves"])
        rmd = _median_curve(m["value_curves"])
        rand_means.append(rm)
        rand_medians.append(rmd)
        ax1.plot(time_axis, rm, color="0.8", linewidth=1.0, alpha=0.8)
    rand_means = np.asarray(rand_means)
    rand_medians = np.asarray(rand_medians)
    ax1.plot(time_axis, np.nanmean(rand_means, axis=0), color="k", linewidth=2.0, label="Model Random Sensors (mean)")
    ax1.plot(time_axis, np.nanmean(rand_medians, axis=0), color="k", linewidth=2.0, linestyle="--", label="Model Random Sensors (median)")
    ax1.set_title("Value Function Comparison")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Q(s, a)")
    ax1.grid(alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(outdir / "compare_value_vs_time.png", dpi=180)

    # Deviation vs time
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    top_mean_d = _mean_curve(top_metrics["deviation_curves"])
    top_median_d = _median_curve(top_metrics["deviation_curves"])
    ax2.plot(time_axis, top_mean_d, color="tab:blue", linewidth=2.2, label="Model Top Sensors (mean)")
    ax2.plot(time_axis, top_median_d, color="tab:blue", linewidth=2.2, linestyle="--", label="Model Top Sensors (median)")

    rand_means_d = []
    rand_medians_d = []
    for m in random_set_metrics:
        rm = _mean_curve(m["deviation_curves"])
        rmd = _median_curve(m["deviation_curves"])
        rand_means_d.append(rm)
        rand_medians_d.append(rmd)
        ax2.plot(time_axis, rm, color="0.8", linewidth=1.0, alpha=0.8)
    rand_means_d = np.asarray(rand_means_d)
    rand_medians_d = np.asarray(rand_medians_d)
    ax2.plot(time_axis, np.nanmean(rand_means_d, axis=0), color="k", linewidth=2.0, label="Model Random Sensors (mean)")
    ax2.plot(time_axis, np.nanmean(rand_medians_d, axis=0), color="k", linewidth=2.0, linestyle="--", label="Model Random Sensors (median)")
    ax2.set_title("State Deviation Comparison")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("||u - u_target||")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(outdir / "compare_deviation_vs_time.png", dpi=180)

    # Stabilization comparison
    fig3, ax3 = plt.subplots(figsize=(11, 6))
    top_x = np.arange(1, stable_top_time.shape[0] + 1)
    ax3.scatter(top_x, stable_top_time, color="tab:blue", s=40, alpha=0.9, label="Model Top Sensors (per eval)")
    if np.any(~np.isnan(stable_top_time)):
        ax3.axhline(np.nanmean(stable_top_time), color="tab:blue", linewidth=2.0, label="Top mean")
        ax3.axhline(np.nanmedian(stable_top_time), color="tab:blue", linewidth=2.0, linestyle="--", label="Top median")

    set_x = np.arange(1, stable_random_time_by_set.shape[0] + 1)
    ax3.scatter(set_x, stable_random_time_by_set, color="0.7", s=40, alpha=0.9, label="Model Random Sensors (per set)")
    if np.any(~np.isnan(stable_random_time_by_set)):
        ax3.axhline(np.nanmean(stable_random_time_by_set), color="k", linewidth=2.0, label="Random mean")
        ax3.axhline(np.nanmedian(stable_random_time_by_set), color="k", linewidth=2.0, linestyle="--", label="Random median")

    ax3.set_title("Time to Stabilize Comparison")
    ax3.set_xlabel("Index (top: eval, random: set)")
    ax3.set_ylabel("Time")
    ax3.grid(alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(outdir / "compare_stabilization.png", dpi=180)

    if not no_show:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = np.loadtxt(args.x_file)
    u_target = np.loadtxt(args.target_file)
    init_states = np.loadtxt(args.init_file)
    if init_states.ndim == 1:
        init_states = init_states.reshape(1, -1)
    if x.ndim != 1:
        raise ValueError("x grid must be 1D")
    if x.size != args.state_dim:
        raise ValueError(f"state_dim mismatch: x has {x.size}, --state-dim is {args.state_dim}")
    if init_states.shape[1] != args.state_dim:
        raise ValueError(
            f"INIT width mismatch: INIT has {init_states.shape[1]}, expected {args.state_dim}"
        )
    if u_target.shape[0] != args.state_dim:
        raise ValueError(
            f"Target width mismatch: target has {u_target.shape[0]}, expected {args.state_dim}"
        )

    episode = pick_checkpoint_episode(args.models_dir, args.episode)
    outdir = Path(args.outdir) if args.outdir else Path(f"./COMPARE_top{args.top_k}_vs_random{args.top_k}_ep{episode}")
    outdir.mkdir(parents=True, exist_ok=True)

    actor = model.Actor(args.state_dim, args.a_dim, args.a_max).to(device)
    critic = model.Critic(args.state_dim, args.a_dim).to(device)
    actor.load_state_dict(torch.load(os.path.join(args.models_dir, f"{episode}_actor.pt"), map_location=device))
    critic.load_state_dict(torch.load(os.path.join(args.models_dir, f"{episode}_critic.pt"), map_location=device))
    actor.eval()
    critic.eval()

    if args.top_sensor_json is not None:
        top_sensor_indices = load_sensor_indices_json(args.top_sensor_json, args.state_dim)
        if len(top_sensor_indices) < 2:
            raise ValueError("Need at least two top sensors for interpolation.")
        if len(top_sensor_indices) != args.top_k:
            raise ValueError(
                f"Top sensor json contains {len(top_sensor_indices)} sensors but --top-k is {args.top_k}. "
                "Please use a top-sensor file with matching size or set --top-k accordingly."
            )
    else:
        if args.importance_json is None:
            raise ValueError("Provide either --top-sensor-json or --importance-json.")
        top_sensor_indices = load_top_sensor_indices(args.importance_json, args.top_k, args.state_dim)
    k_select = len(top_sensor_indices)
    random_sensor_sets = make_random_sensor_sets(
        state_dim=args.state_dim,
        set_size=k_select,
        num_sets=args.num_random_sets,
        rng=rng,
    )
    init_rows = rng.integers(0, init_states.shape[0], size=args.num_evals, endpoint=False)
    # Balance sample counts: top condition uses the same total rollout count
    # as all random sets combined (num_random_sets * num_evals).
    top_init_rows = np.tile(init_rows, len(random_sensor_sets))

    top_metrics = run_condition(
        actor=actor,
        critic=critic,
        x=x,
        u_target=u_target,
        init_states=init_states,
        init_rows=top_init_rows,
        sensor_idx=top_sensor_indices,
        a_dim=args.a_dim,
        max_steps=args.max_steps,
        stabilize_threshold=args.stabilize_threshold,
        stabilize_hold=args.stabilize_hold,
        domain_length=args.domain_length,
        device=device,
    )

    random_metrics: List[Dict[str, np.ndarray]] = []
    for sset in random_sensor_sets:
        m = run_condition(
            actor=actor,
            critic=critic,
            x=x,
            u_target=u_target,
            init_states=init_states,
            init_rows=init_rows,
            sensor_idx=sset,
            a_dim=args.a_dim,
            max_steps=args.max_steps,
            stabilize_threshold=args.stabilize_threshold,
            stabilize_hold=args.stabilize_hold,
            domain_length=args.domain_length,
            device=device,
        )
        random_metrics.append(m)

    dt = KS(L=args.domain_length, N=args.state_dim, a_dim=args.a_dim).dt
    time_axis = np.arange(args.max_steps, dtype=np.float32) * dt
    stable_top_time = top_metrics["stable_steps"] * dt
    stable_random_time_by_set = []
    for m in random_metrics:
        s = m["stable_steps"]
        if np.any(~np.isnan(s)):
            stable_random_time_by_set.append(float(np.nanmean(s)))
        else:
            stable_random_time_by_set.append(np.nan)
    stable_random_time_by_set = np.asarray(stable_random_time_by_set, dtype=np.float32) * dt

    # Save artifacts
    (outdir / "top_sensor_indices.json").write_text(json.dumps(top_sensor_indices, indent=2), encoding="utf-8")
    (outdir / "random_sensor_sets.json").write_text(json.dumps(random_sensor_sets, indent=2), encoding="utf-8")
    run_config = {
        "labels": ["model_top_sensors", "model_random_sensors"],
        "models_dir": args.models_dir,
        "episode": episode,
        "importance_json": args.importance_json,
        "top_sensor_json": args.top_sensor_json,
        "top_k": k_select,
        "state_dim": args.state_dim,
        "a_dim": args.a_dim,
        "a_max": args.a_max,
        "num_evals": args.num_evals,
        "num_random_sets": len(random_sensor_sets),
        "max_steps": args.max_steps,
        "domain_length": args.domain_length,
        "stabilize_threshold": args.stabilize_threshold,
        "stabilize_hold": args.stabilize_hold,
        "seed": args.seed,
        "init_rows_base": init_rows.tolist(),
        "init_rows_top": top_init_rows.tolist(),
        "top_total_rollouts": int(top_init_rows.shape[0]),
        "random_total_rollouts": int(len(random_sensor_sets) * args.num_evals),
        "interpolation": "periodic_linear",
    }
    (outdir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    np.savez_compressed(
        outdir / "metrics_top.npz",
        value_curves=top_metrics["value_curves"],
        deviation_curves=top_metrics["deviation_curves"],
        stable_steps=top_metrics["stable_steps"],
    )

    random_value_curves = np.asarray([m["value_curves"] for m in random_metrics], dtype=np.float32)
    random_deviation_curves = np.asarray([m["deviation_curves"] for m in random_metrics], dtype=np.float32)
    random_stable_steps = np.asarray([m["stable_steps"] for m in random_metrics], dtype=np.float32)
    np.savez_compressed(
        outdir / "metrics_random.npz",
        value_curves=random_value_curves,
        deviation_curves=random_deviation_curves,
        stable_steps=random_stable_steps,
    )

    summary = {
        "labels": {"top": "model_top_sensors", "random": "model_random_sensors"},
        "top_sensor_indices": top_sensor_indices,
        "top": {
            "mean_final_deviation": float(np.nanmean(top_metrics["deviation_curves"][:, -1])),
            "median_final_deviation": float(np.nanmedian(top_metrics["deviation_curves"][:, -1])),
            "mean_stabilize_time": float(np.nanmean(stable_top_time)) if np.any(~np.isnan(stable_top_time)) else None,
            "median_stabilize_time": float(np.nanmedian(stable_top_time)) if np.any(~np.isnan(stable_top_time)) else None,
        },
        "random": {
            "mean_final_deviation_over_sets": float(np.nanmean(random_deviation_curves[:, :, -1])),
            "median_final_deviation_over_sets": float(np.nanmedian(random_deviation_curves[:, :, -1])),
            "mean_stabilize_time_over_sets": float(np.nanmean(stable_random_time_by_set))
            if np.any(~np.isnan(stable_random_time_by_set))
            else None,
            "median_stabilize_time_over_sets": float(np.nanmedian(stable_random_time_by_set))
            if np.any(~np.isnan(stable_random_time_by_set))
            else None,
        },
    }
    (outdir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_comparison(
        time_axis=time_axis,
        top_metrics=top_metrics,
        random_set_metrics=random_metrics,
        stable_top_time=stable_top_time,
        stable_random_time_by_set=stable_random_time_by_set,
        outdir=outdir,
        no_show=args.no_show,
    )

    print(f"Loaded checkpoint episode: {episode}")
    print(f"Saved comparison artifacts to: {outdir}")
    print(f"Model Top Sensors (top-{k_select}): {top_sensor_indices}")
    print(f"Model Random Sensors: {len(random_sensor_sets)} sets")


if __name__ == "__main__":
    main()
