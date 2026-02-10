from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from KS import KS
import model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-model rollout evaluation with shared INIT rows and physics-based metrics."
    )
    parser.add_argument("--models-spec", type=str, required=True, help="Path to models spec JSON.")
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--domain-length", type=float, default=22.0)
    parser.add_argument("--target-file", type=str, default="u3.dat")
    parser.add_argument("--x-file", type=str, default="x.dat")
    parser.add_argument("--init-file", type=str, default="INIT.dat")
    parser.add_argument("--num-evals", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--dwell-time", type=float, required=True, help="Stabilization dwell time in seconds.")
    parser.add_argument("--epsilon-beta", type=float, default=0.10)
    parser.add_argument("--dt-expected", type=float, default=0.05)
    parser.add_argument("--final-window-frac", type=float, default=0.2)
    parser.add_argument("--final-window-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="./EVAL_multi_model")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--plot-critic-diagnostic", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def pick_checkpoint_episode(models_dir: str, requested_episode: Optional[int], need_critic: bool) -> int:
    if requested_episode is not None:
        actor_path = os.path.join(models_dir, f"{requested_episode}_actor.pt")
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"Missing actor checkpoint: {actor_path}")
        if need_critic:
            critic_path = os.path.join(models_dir, f"{requested_episode}_critic.pt")
            if not os.path.exists(critic_path):
                raise FileNotFoundError(f"Missing critic checkpoint: {critic_path}")
        return requested_episode

    actor_ckpts = [
        f for f in os.listdir(models_dir)
        if f.endswith("_actor.pt") and not f.endswith("_target_actor.pt")
    ]
    if not actor_ckpts:
        raise FileNotFoundError(f"No actor checkpoints found in {models_dir}")
    episode = max(int(f.split("_")[0]) for f in actor_ckpts)
    if need_critic:
        critic_path = os.path.join(models_dir, f"{episode}_critic.pt")
        if not os.path.exists(critic_path):
            raise FileNotFoundError(f"Missing critic checkpoint: {critic_path}")
    return episode


def sanitize_key(name: str) -> str:
    out = []
    for c in name:
        out.append(c if c.isalnum() else "_")
    key = "".join(out).strip("_")
    return key or "model"


def load_models_spec(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"models-spec file not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "models" not in data:
            raise ValueError("models-spec dict must contain key 'models'.")
        data = data["models"]
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("models-spec must be a non-empty list.")

    out: List[Dict[str, Any]] = []
    for i, m in enumerate(data):
        if not isinstance(m, dict):
            raise ValueError(f"models-spec entry #{i} must be an object.")
        for req in ["name", "role", "models_dir", "s_dim", "a_dim", "a_max"]:
            if req not in m:
                raise ValueError(f"models-spec entry '{m}' missing required field '{req}'.")
        role = str(m["role"])
        if role not in {"bucci8", "shap8", "full64", "other"}:
            raise ValueError(f"Invalid role '{role}'. Use bucci8, shap8, full64, or other.")
        if role in {"bucci8", "shap8"} and "sensor_indices" not in m:
            raise ValueError(f"Role '{role}' requires explicit sensor_indices.")
        out.append(m)
    return out


def resolve_sensor_indices(spec: Dict[str, Any], state_dim: int) -> np.ndarray:
    s_dim = int(spec["s_dim"])
    raw = spec.get("sensor_indices", None)
    if raw is None:
        step = state_dim // s_dim
        if step <= 0:
            raise ValueError(
                f"Invalid equispaced step for model '{spec['name']}': state_dim={state_dim}, s_dim={s_dim}"
            )
        idx = np.arange(0, state_dim, step, dtype=np.int64)
    else:
        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError(f"sensor_indices for model '{spec['name']}' must be a non-empty list.")
        idx = np.asarray([int(v) for v in raw], dtype=np.int64)

    if idx.size != s_dim:
        raise ValueError(
            f"sensor_indices size mismatch for model '{spec['name']}': got {idx.size}, expected s_dim={s_dim}."
        )
    if np.unique(idx).size != idx.size:
        raise ValueError(f"sensor_indices contains duplicates for model '{spec['name']}'.")
    if np.any(idx < 0) or np.any(idx >= state_dim):
        raise ValueError(f"sensor_indices out of bounds for model '{spec['name']}'.")
    return idx


def first_stable_step(error_curve: np.ndarray, epsilon: float, dwell_steps: int) -> Optional[int]:
    n = error_curve.shape[0]
    if dwell_steps <= 1:
        hits = np.where(error_curve <= epsilon)[0]
        return int(hits[0]) if hits.size > 0 else None
    if dwell_steps > n:
        return None
    for t in range(0, n - dwell_steps + 1):
        if np.all(error_curve[t:t + dwell_steps] <= epsilon):
            return t
    return None


def _sample_std(x: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def _final_window_start(max_steps: int, final_window_frac: float, final_window_steps: Optional[int]) -> int:
    if final_window_steps is not None:
        if final_window_steps <= 0 or final_window_steps > max_steps:
            raise ValueError(
                f"final-window-steps must be in [1, {max_steps}], got {final_window_steps}"
            )
        n_window = final_window_steps
    else:
        if not (0.0 < final_window_frac <= 1.0):
            raise ValueError(f"final-window-frac must be in (0,1], got {final_window_frac}")
        n_window = max(1, int(np.ceil(final_window_frac * max_steps)))
    return max_steps - n_window


def evaluate_model(
    spec: Dict[str, Any],
    x: np.ndarray,
    u_target: np.ndarray,
    init_states: np.ndarray,
    init_rows: np.ndarray,
    max_steps: int,
    domain_length: float,
    epsilon: float,
    dwell_time: float,
    final_start: int,
    dt_expected: float,
    plot_critic_diagnostic: bool,
    device: torch.device,
) -> Dict[str, Any]:
    name = str(spec["name"])
    role = str(spec["role"])
    models_dir = str(spec["models_dir"])
    s_dim = int(spec["s_dim"])
    a_dim = int(spec["a_dim"])
    a_max = float(spec["a_max"])
    sensor_indices = resolve_sensor_indices(spec, state_dim=x.size)

    episode = pick_checkpoint_episode(
        models_dir=models_dir,
        requested_episode=spec.get("episode", None),
        need_critic=plot_critic_diagnostic,
    )

    actor = model.Actor(s_dim, a_dim, a_max).to(device)
    actor_path = os.path.join(models_dir, f"{episode}_actor.pt")
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    critic = None
    if plot_critic_diagnostic:
        critic = model.Critic(s_dim, a_dim).to(device)
        critic_path = os.path.join(models_dir, f"{episode}_critic.pt")
        critic.load_state_dict(torch.load(critic_path, map_location=device))
        critic.eval()

    ks = KS(L=domain_length, N=x.size, a_dim=a_dim)
    dt = float(ks.dt)
    if abs(dt - dt_expected) > 1e-12:
        raise ValueError(
            f"dt mismatch for model '{name}': KS dt={dt} but dt_expected={dt_expected}"
        )
    dwell_steps = int(np.ceil(dwell_time / dt))
    if dwell_steps <= 0:
        raise ValueError(f"dwell-time must yield positive dwell steps, got dwell_time={dwell_time}, dt={dt}")

    n_eval = init_rows.shape[0]
    error_curves = np.zeros((n_eval, max_steps), dtype=np.float32)
    action_norm2_curves = np.zeros((n_eval, max_steps), dtype=np.float32)
    final_error = np.zeros((n_eval,), dtype=np.float32)
    t_stab = np.full((n_eval,), np.nan, dtype=np.float32)
    value_curves = np.zeros((n_eval, max_steps), dtype=np.float32) if plot_critic_diagnostic else None

    with torch.no_grad():
        for k in range(n_eval):
            obs = np.float32(init_states[int(init_rows[k])].copy())
            for t in range(max_steps):
                state = np.float32(obs[sensor_indices])
                st = torch.from_numpy(state).to(device).unsqueeze(0)
                at = actor(st)
                action = at.squeeze(0).cpu().numpy()
                action_norm2_curves[k, t] = float(np.sum(action ** 2))

                if critic is not None and value_curves is not None:
                    qt = critic(st, at)
                    value_curves[k, t] = float(qt.item())

                obs = ks.advance(obs, action)
                error_curves[k, t] = float(np.linalg.norm(obs - u_target))

            final_error[k] = float(np.mean(error_curves[k, final_start:]))
            hit = first_stable_step(error_curves[k], epsilon=epsilon, dwell_steps=dwell_steps)
            if hit is not None:
                t_stab[k] = np.float32(hit * dt)

    if not np.all(np.isfinite(error_curves)):
        raise ValueError(f"Non-finite error values detected for model '{name}'.")

    effort = np.sum(action_norm2_curves, axis=1) * dt
    horizon_time = max_steps * dt
    success_mask = ~np.isnan(t_stab)
    success_rate = float(np.mean(success_mask))

    summary = {
        "name": name,
        "role": role,
        "episode": int(episode),
        "s_dim": s_dim,
        "a_dim": a_dim,
        "a_max": a_max,
        "sensor_indices": sensor_indices.tolist(),
        "dt": dt,
        "dwell_steps": int(dwell_steps),
        "metric1_final_error": {
            "mean": float(np.mean(final_error)),
            "std": _sample_std(final_error),
            "median": float(np.median(final_error)),
        },
        "metric2_stabilization": {
            "success_rate": success_rate,
            "mean_t_stab_success": float(np.nanmean(t_stab)) if np.any(success_mask) else None,
            "median_t_stab_success": float(np.nanmedian(t_stab)) if np.any(success_mask) else None,
            "num_success": int(np.sum(success_mask)),
            "num_total": int(n_eval),
        },
        "metric3_control_effort": {
            "mean": float(np.mean(effort)),
            "std": _sample_std(effort),
            "median": float(np.median(effort)),
            "mean_normalized_over_T": float(np.mean(effort / horizon_time)),
        },
    }

    return {
        "name": name,
        "role": role,
        "episode": int(episode),
        "dt": dt,
        "sensor_indices": sensor_indices,
        "error_curves": error_curves,
        "action_norm2_curves": action_norm2_curves,
        "final_error": final_error,
        "t_stab": t_stab,
        "effort": effort.astype(np.float32),
        "value_curves": value_curves,
        "summary": summary,
    }


def _ensure_outdir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _set_box_axis_limits(bp: Dict[str, Any], ax: plt.Axes, horizontal: bool = False) -> None:
    whiskers = bp.get("whiskers", [])
    if not whiskers:
        return

    vals: List[float] = []
    for w in whiskers:
        arr = w.get_xdata() if horizontal else w.get_ydata()
        vals.extend([float(v) for v in arr if np.isfinite(v)])

    if not vals:
        return

    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if lo == hi:
        pad = max(1e-6, abs(lo) * 0.05 + 1e-6)
    else:
        pad = 0.05 * (hi - lo)

    if horizontal:
        ax.set_xlim(lo - pad, hi + pad)
    else:
        ax.set_ylim(lo - pad, hi + pad)


def _plot_error_vs_time(
    outdir: Path,
    time_axis: np.ndarray,
    model_outputs: List[Dict[str, Any]],
    epsilon: float,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(model_outputs):
        curves = m["error_curves"]
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        c = cmap(i % 10)
        ax.plot(time_axis, mean_curve, color=c, linewidth=2.2, label=m["name"])
        ax.fill_between(
            time_axis,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=c,
            alpha=0.2,
        )
    ax.axhline(epsilon, color="k", linestyle=":", linewidth=1.8, label=f"epsilon={epsilon:.4g}")
    ax.set_title("Metric 1.1 Mean Tracking Error vs Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("||v - E3||_2")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "plot_1_1_error_vs_time.png", dpi=180)
    plt.close(fig)


def _plot_final_error_boxplot(outdir: Path, model_outputs: List[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [m["final_error"] for m in model_outputs]
    labels = [m["name"] for m in model_outputs]
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, showfliers=False)
    _set_box_axis_limits(bp, ax, horizontal=False)
    ax.set_title("Metric 1.2 Final Error Distribution")
    ax.set_ylabel("final-window mean ||v - E3||_2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "plot_1_2_final_error_boxplot.png", dpi=180)
    plt.close(fig)


def _plot_success_rate(outdir: Path, model_outputs: List[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [m["name"] for m in model_outputs]
    sr = [float(np.mean(~np.isnan(m["t_stab"]))) for m in model_outputs]
    ax.bar(np.arange(len(labels)), sr, color="0.4")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Metric 2.1 Stabilization Success Rate")
    ax.set_ylabel("Success Rate")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "plot_2_1_success_rate.png", dpi=180)
    plt.close(fig)


def _plot_tstab_success_only(outdir: Path, model_outputs: List[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    labels: List[str] = []
    data: List[np.ndarray] = []
    no_success: List[str] = []

    for m in model_outputs:
        s = m["t_stab"]
        s_ok = s[~np.isnan(s)]
        if s_ok.size == 0:
            no_success.append(m["name"])
            continue
        labels.append(m["name"])
        data.append(s_ok)

    if data:
        bp = ax.boxplot(data, tick_labels=labels, showmeans=True, showfliers=False)
        _set_box_axis_limits(bp, ax, horizontal=False)
    else:
        ax.text(0.5, 0.5, "No successful stabilizations in any model.", ha="center", va="center")
        ax.set_xticks([])

    if no_success:
        ax.text(
            0.02,
            0.98,
            "No-success models: " + ", ".join(no_success),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
        )

    ax.set_title("Metric 2.2 Time-to-Stabilize (Successful Runs Only)")
    ax.set_ylabel("t_stab (time)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "plot_2_2_tstab_boxplot_success_only.png", dpi=180)
    plt.close(fig)


def _plot_stabilization_trajectories(
    outdir: Path,
    time_axis: np.ndarray,
    model_outputs: List[Dict[str, Any]],
    epsilon: float,
    dwell_time: float,
) -> None:
    n_models = len(model_outputs)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, max(3.6, 3.4 * n_models)), sharex=True)
    if n_models == 1:
        axes = [axes]

    for ax, m in zip(axes, model_outputs):
        err = m["error_curves"]
        t_stab = m["t_stab"]

        # Show a small set of representative trajectories for readability.
        max_show = min(8, err.shape[0])
        show_idx = np.arange(max_show)
        for i in show_idx:
            ax.plot(time_axis, err[i], color="0.75", linewidth=1.0, alpha=0.9)

        mean_curve = np.mean(err, axis=0)
        ax.plot(time_axis, mean_curve, color="k", linewidth=2.0, label="mean error")
        ax.axhline(epsilon, color="tab:red", linestyle=":", linewidth=1.7, label="epsilon")

        success_idx = np.where(~np.isnan(t_stab))[0]
        if success_idx.size > 0:
            k = int(success_idx[0])
            t0 = float(t_stab[k])
            t1 = min(float(time_axis[-1]), t0 + float(dwell_time))
            ax.axvspan(t0, t1, color="tab:green", alpha=0.16, label="example dwell window")
            ax.plot(time_axis, err[k], color="tab:green", linewidth=1.4, alpha=0.95, label="example successful run")
        else:
            ax.text(
                0.98,
                0.95,
                "No successful run",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
            )

        ax.set_title(f"Metric 2.3 Example Stabilization Trajectories - {m['name']}")
        ax.set_ylabel("||v - E3||_2")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(outdir / "plot_2_3_example_stabilization_trajectories.png", dpi=180)
    plt.close(fig)


def _plot_effort_vs_error(outdir: Path, model_outputs: List[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(model_outputs):
        ax.scatter(
            m["effort"],
            m["final_error"],
            s=26,
            alpha=0.8,
            color=cmap(i % 10),
            label=m["name"],
        )
    ax.set_title("Metric 3.1 Control Effort vs Final Error")
    ax.set_xlabel("J_u")
    ax.set_ylabel("final-window mean ||v - E3||_2")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "plot_3_1_effort_vs_final_error.png", dpi=180)
    plt.close(fig)


def _plot_effort_boxplot(outdir: Path, model_outputs: List[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [m["effort"] for m in model_outputs]
    labels = [m["name"] for m in model_outputs]
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, showfliers=False)
    _set_box_axis_limits(bp, ax, horizontal=False)
    ax.set_title("Metric 3.2 Control Effort Distribution")
    ax.set_ylabel("J_u")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "plot_3_2_effort_boxplot.png", dpi=180)
    plt.close(fig)


def _plot_control_magnitude_vs_time(
    outdir: Path,
    time_axis: np.ndarray,
    model_outputs: List[Dict[str, Any]],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(model_outputs):
        # action_norm2_curves stores ||u||_2^2; convert to ||u||_2 for this plot.
        mag = np.sqrt(np.maximum(0.0, m["action_norm2_curves"]))
        mean_curve = np.mean(mag, axis=0)
        std_curve = np.std(mag, axis=0)
        c = cmap(i % 10)
        ax.plot(time_axis, mean_curve, color=c, linewidth=2.2, label=m["name"])
        ax.fill_between(time_axis, mean_curve - std_curve, mean_curve + std_curve, color=c, alpha=0.2)

    ax.set_title("Metric 3.3 Mean Control Magnitude vs Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("||u(t)||_2")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "plot_3_3_control_magnitude_vs_time.png", dpi=180)
    plt.close(fig)


def _plot_paired_final_error_diff(outdir: Path, model_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    bucci = next((m for m in model_outputs if m["role"] == "bucci8"), None)
    shap = next((m for m in model_outputs if m["role"] == "shap8"), None)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    if bucci is None or shap is None:
        msg = "Paired-diff unavailable: need both bucci8 and shap8 models."
        axes[0].text(0.5, 0.5, msg, ha="center", va="center")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].set_visible(False)
        fig.savefig(outdir / "plot_final_error_paired_diff.png", dpi=180)
        plt.close(fig)
        return {"available": False, "message": msg}

    if bucci["final_error"].shape[0] != shap["final_error"].shape[0]:
        raise ValueError("Paired-diff requires same number of rollouts for bucci8 and shap8.")

    delta = shap["final_error"] - bucci["final_error"]
    frac_better = float(np.mean(delta < 0.0))

    axes[0].hist(delta, bins=20, color="0.7", edgecolor="0.3")
    axes[0].axvline(0.0, color="k", linestyle=":", linewidth=1.6)
    axes[0].axvline(float(np.mean(delta)), color="tab:red", linewidth=1.8, label="mean(Delta)")
    axes[0].set_title("Paired Final Error Difference: Delta = SHAP - Bucci")
    axes[0].set_xlabel("Delta")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[0].text(
        0.98,
        0.95,
        f"P(Delta < 0) = {frac_better:.3f}",
        ha="right",
        va="top",
        transform=axes[0].transAxes,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )

    bp = axes[1].boxplot([delta], vert=False, tick_labels=["SHAP - Bucci"], showmeans=True, showfliers=False)
    _set_box_axis_limits(bp, axes[1], horizontal=True)
    axes[1].axvline(0.0, color="k", linestyle=":", linewidth=1.6)
    axes[1].set_xlabel("Delta")
    axes[1].grid(alpha=0.3)

    fig.savefig(outdir / "plot_final_error_paired_diff.png", dpi=180)
    plt.close(fig)
    return {
        "available": True,
        "delta_mean": float(np.mean(delta)),
        "delta_std": _sample_std(delta),
        "delta_median": float(np.median(delta)),
        "fraction_delta_lt_0": frac_better,
    }


def _plot_critic_diagnostic(outdir: Path, time_axis: np.ndarray, model_outputs: List[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")
    plotted = 0
    for i, m in enumerate(model_outputs):
        if m["value_curves"] is None:
            continue
        v = m["value_curves"]
        mean_curve = np.mean(v, axis=0)
        med_curve = np.median(v, axis=0)
        c = cmap(i % 10)
        ax.plot(time_axis, mean_curve, color=c, linewidth=2.0, label=f"{m['name']} mean")
        ax.plot(time_axis, med_curve, color=c, linewidth=1.8, linestyle="--", label=f"{m['name']} median")
        plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "No critic diagnostics available.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel("Time")
        ax.set_ylabel("Q(s,a)")
        ax.set_title("Critic Value Function Evolution (Diagnostic Only)")
        ax.grid(alpha=0.3)
        ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "plot_diag_value_vs_time.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.num_evals <= 0:
        raise ValueError("num-evals must be positive.")
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive.")
    if args.domain_length <= 0:
        raise ValueError("domain-length must be positive.")
    if args.dwell_time <= 0:
        raise ValueError("dwell-time must be positive.")
    if not (0.0 < args.epsilon_beta <= 1.0):
        raise ValueError("epsilon-beta must be in (0, 1].")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    model_specs = load_models_spec(args.models_spec)
    if not args.include_reference:
        model_specs = [m for m in model_specs if str(m["role"]) != "full64"]
        if not model_specs:
            raise ValueError("No models left after excluding reference models. Use --include-reference.")

    x = np.loadtxt(args.x_file)
    u_target = np.loadtxt(args.target_file)
    init_states = np.loadtxt(args.init_file)
    if init_states.ndim == 1:
        init_states = init_states.reshape(1, -1)

    if x.ndim != 1:
        raise ValueError(f"x-file must be 1D, got shape {x.shape}")
    if x.size != args.state_dim:
        raise ValueError(f"x size ({x.size}) must match --state-dim ({args.state_dim})")
    if u_target.ndim != 1 or u_target.size != args.state_dim:
        raise ValueError(
            f"target-file must be 1D length {args.state_dim}, got shape {u_target.shape}"
        )
    if init_states.ndim != 2 or init_states.shape[1] != args.state_dim:
        raise ValueError(
            f"init-file must have width {args.state_dim}, got shape {init_states.shape}"
        )

    epsilon = float(args.epsilon_beta * np.linalg.norm(u_target))
    epsilon_source = "target_relative_method3"

    final_start = _final_window_start(
        max_steps=args.max_steps,
        final_window_frac=args.final_window_frac,
        final_window_steps=args.final_window_steps,
    )

    init_rows = rng.integers(0, init_states.shape[0], size=args.num_evals, endpoint=False, dtype=np.int64)

    outputs: List[Dict[str, Any]] = []
    for spec in model_specs:
        out = evaluate_model(
            spec=spec,
            x=x,
            u_target=u_target,
            init_states=init_states,
            init_rows=init_rows,
            max_steps=args.max_steps,
            domain_length=args.domain_length,
            epsilon=epsilon,
            dwell_time=args.dwell_time,
            final_start=final_start,
            dt_expected=args.dt_expected,
            plot_critic_diagnostic=args.plot_critic_diagnostic,
            device=device,
        )
        outputs.append(out)

    if not outputs:
        raise ValueError("No models evaluated.")

    dt_used = float(outputs[0]["dt"])
    time_axis = np.arange(args.max_steps, dtype=np.float32) * dt_used
    outdir = _ensure_outdir(args.outdir)

    _plot_error_vs_time(outdir, time_axis, outputs, epsilon)
    _plot_final_error_boxplot(outdir, outputs)
    _plot_success_rate(outdir, outputs)
    _plot_tstab_success_only(outdir, outputs)
    _plot_stabilization_trajectories(outdir, time_axis, outputs, epsilon=epsilon, dwell_time=args.dwell_time)
    _plot_effort_vs_error(outdir, outputs)
    _plot_effort_boxplot(outdir, outputs)
    _plot_control_magnitude_vs_time(outdir, time_axis, outputs)
    paired_meta = _plot_paired_final_error_diff(outdir, outputs)
    if args.plot_critic_diagnostic:
        _plot_critic_diagnostic(outdir, time_axis, outputs)

    run_config = {
        "models_spec_path": args.models_spec,
        "models_evaluated": [
            {
                "name": o["name"],
                "role": o["role"],
                "episode": o["episode"],
                "sensor_indices": o["sensor_indices"].tolist(),
            }
            for o in outputs
        ],
        "state_dim": args.state_dim,
        "num_evals": args.num_evals,
        "max_steps": args.max_steps,
        "domain_length": args.domain_length,
        "target_file": args.target_file,
        "x_file": args.x_file,
        "init_file": args.init_file,
        "seed": args.seed,
        "init_rows": init_rows.tolist(),
        "dt_expected": args.dt_expected,
        "dt_used": dt_used,
        "epsilon_method": epsilon_source,
        "epsilon_beta": args.epsilon_beta,
        "epsilon_value": epsilon,
        "error_norm": "l2_full_state",
        "dwell_time": args.dwell_time,
        "dwell_steps": int(np.ceil(args.dwell_time / dt_used)),
        "final_window_start_idx": int(final_start),
        "final_window_steps": int(args.max_steps - final_start),
        "include_reference": bool(args.include_reference),
        "plot_critic_diagnostic": bool(args.plot_critic_diagnostic),
        "paired_diff": paired_meta,
    }
    (outdir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    summary = {
        "epsilon": epsilon,
        "dt": dt_used,
        "models": {o["name"]: o["summary"] for o in outputs},
        "paired_diff": paired_meta,
    }
    (outdir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    arrays: Dict[str, Any] = {
        "init_rows": init_rows,
        "time_axis": time_axis,
        "epsilon": np.float32(epsilon),
        "dt": np.float32(dt_used),
    }
    for i, o in enumerate(outputs):
        key = f"{i}_{sanitize_key(o['name'])}"
        arrays[f"{key}__role"] = np.asarray([o["role"]], dtype=object)
        arrays[f"{key}__error_curves"] = o["error_curves"]
        arrays[f"{key}__action_norm2_curves"] = o["action_norm2_curves"]
        arrays[f"{key}__final_error"] = o["final_error"]
        arrays[f"{key}__t_stab"] = o["t_stab"]
        arrays[f"{key}__effort"] = o["effort"]
        arrays[f"{key}__sensor_indices"] = o["sensor_indices"]
        arrays[f"{key}__episode"] = np.int64(o["episode"])
        if o["value_curves"] is not None:
            arrays[f"{key}__value_curves"] = o["value_curves"]

    np.savez_compressed(outdir / "metrics_arrays.npz", **arrays)

    print(f"Saved run config: {outdir / 'run_config.json'}")
    print(f"Saved summary metrics: {outdir / 'metrics_summary.json'}")
    print(f"Saved arrays: {outdir / 'metrics_arrays.npz'}")
    print(f"Saved plots in: {outdir}")
    print(f"Epsilon (Method 3): {epsilon:.6f} = {args.epsilon_beta:.3f} * ||E3||_2")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
