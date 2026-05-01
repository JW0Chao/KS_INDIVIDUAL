from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
SENSOR_COUNT_RESULTS_DIR = ROOT / "studies" / "sensor_count" / "results"
STUDY_PROTOCOL_DIR = ROOT / "studies" / "controller_protocol"
DEFAULT_INIT_SPLIT_FILE = STUDY_PROTOCOL_DIR / "manifests" / "controller_setup_split.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ks_control.ks import KS
from ks_control import model
from ks_control.evaluation_helpers import first_stable_step, pick_checkpoint_episode, resolve_sensor_indices
from ks_control.study_protocol import (
    compute_target_relative_epsilon,
    integrated_control_effort,
    integrated_error,
    load_or_create_split_manifest,
    summarize_run_rows,
    summarize_setup_runs,
)

plt.rcParams.update(
    {
        "font.size": 16,
        "font.weight": "normal",
        "axes.labelsize": 22,
        "axes.labelweight": "normal",
        "axes.titlesize": 24,
        "axes.titleweight": "normal",
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 1.6,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "legend.fontsize": 14,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "white",
        "legend.edgecolor": "0.35",
    }
)

MODEL_K_RE = re.compile(r".*_k(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-model rollout evaluation with shared INIT rows and physics-based metrics."
    )
    parser.add_argument("--models-spec", type=str, required=True, help="Path to models spec JSON.")
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--domain-length", type=float, default=22.0)
    parser.add_argument("--target-file", type=str, default=str(DATA_DIR / "u3.dat"))
    parser.add_argument("--x-file", type=str, default=str(DATA_DIR / "x.dat"))
    parser.add_argument("--init-file", type=str, default=str(DATA_DIR / "INIT.dat"))
    parser.add_argument("--init-split-file", type=str, default=str(DEFAULT_INIT_SPLIT_FILE))
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--train-split-size", type=int, default=20)
    parser.add_argument("--val-split-size", type=int, default=20)
    parser.add_argument("--test-split-size", type=int, default=30)
    parser.add_argument("--split-role", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-evals", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--dwell-time", type=float, required=True, help="Stabilisation dwell time in seconds.")
    parser.add_argument("--epsilon-beta", type=float, default=0.10)
    parser.add_argument(
        "--epsilon-mode",
        type=str,
        default="target_relative",
        choices=["target_relative"],
        help="The uniform-study evaluator uses target-relative epsilon: beta * ||E3||_2.",
    )
    parser.add_argument("--dt-expected", type=float, default=0.05)
    parser.add_argument("--final-window-frac", type=float, default=0.2)
    parser.add_argument("--final-window-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default=str(SENSOR_COUNT_RESULTS_DIR / "evaluation"))
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--plot-critic-diagnostic", action="store_true")
    parser.add_argument(
        "--error-plot-engine",
        type=str,
        default="plotly",
        choices=["plotly", "matplotlib"],
        help="Backend for split error plots (mean/median).",
    )
    parser.add_argument(
        "--error-show-epsilon",
        dest="error_show_epsilon",
        action="store_true",
        help="Show epsilon threshold line in split error plots (default).",
    )
    parser.add_argument(
        "--error-hide-epsilon",
        dest="error_show_epsilon",
        action="store_false",
        help="Hide epsilon threshold line in split error plots.",
    )
    parser.set_defaults(error_show_epsilon=True)
    parser.add_argument(
        "--error-plot-width",
        type=int,
        default=1200,
        help="Split error plot width in pixels (matplotlib size is converted from this).",
    )
    parser.add_argument("--error-plot-height", type=int, default=600, help="Split error plot height in pixels.")
    parser.add_argument("--error-line-width", type=float, default=3.2)
    parser.add_argument("--error-template", type=str, default="plotly_white")
    parser.add_argument("--error-x-label", type=str, default="Time")
    parser.add_argument("--error-y-label", type=str, default="||v - E3||<sub>2</sub>")
    parser.add_argument(
        "--error-title-mean",
        type=str,
        default="Mean Control Error vs Time",
    )
    parser.add_argument(
        "--error-title-median",
        type=str,
        default="Median Control Error vs Time",
    )
    parser.add_argument(
        "--control-trajectory-k",
        type=int,
        default=8,
        help="Plot control trajectories only for models with this sensor count. Set <=0 to disable.",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


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
        if not role:
            raise ValueError("models-spec role must be a non-empty string.")
        out.append(m)
    return out


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


def _compute_uncontrolled_baseline_error_curves(
    x: np.ndarray,
    u_target: np.ndarray,
    init_states: np.ndarray,
    init_rows: np.ndarray,
    max_steps: int,
    domain_length: float,
    a_dim: int,
) -> Tuple[np.ndarray, float]:
    ks = KS(L=domain_length, N=x.size, a_dim=a_dim)
    dt = float(ks.dt)
    n_eval = int(init_rows.size)
    error_curves = np.zeros((n_eval, max_steps), dtype=np.float32)
    zero_action = np.zeros((a_dim,), dtype=np.float32)
    for k in range(n_eval):
        obs = np.float32(init_states[int(init_rows[k])].copy())
        for t in range(max_steps):
            obs = ks.advance(obs, zero_action)
            error_curves[k, t] = float(np.linalg.norm(obs - u_target))
    return error_curves, dt


def _compute_stabilization_from_error_curves(
    error_curves: np.ndarray,
    epsilon: float,
    dwell_steps: int,
    dt: float,
) -> Dict[str, Any]:
    n_eval = int(error_curves.shape[0])
    t_stab = np.full((n_eval,), np.nan, dtype=np.float32)
    for k in range(n_eval):
        hit = first_stable_step(error_curves[k], epsilon=epsilon, dwell_steps=dwell_steps)
        if hit is not None:
            t_stab[k] = np.float32(hit * dt)
    success_mask = ~np.isnan(t_stab)
    summary = {
        "success_rate": float(np.mean(success_mask)),
        "mean_t_stab_success": float(np.nanmean(t_stab)) if np.any(success_mask) else None,
        "median_t_stab_success": float(np.nanmedian(t_stab)) if np.any(success_mask) else None,
        "num_success": int(np.sum(success_mask)),
        "num_total": int(n_eval),
    }
    return {
        "t_stab": t_stab,
        "summary": summary,
    }


def _clone_outputs_with_tstab(
    model_outputs: List[Dict[str, Any]],
    tstab_by_name: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in model_outputs:
        name = str(m["name"])
        if name not in tstab_by_name:
            raise KeyError(f"Missing t_stab for model '{name}'.")
        clone = dict(m)
        clone["t_stab"] = tstab_by_name[name]
        out.append(clone)
    return out


def _epsilon_method_title(method: str) -> str:
    if method == "target_relative":
        return "target-relative epsilon"
    return method


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
    setup_name = str(spec.get("setup_name", name))
    target_name = str(spec.get("target_name", Path(str(spec.get("target_file", "u3.dat"))).stem))
    rl_seed = None if spec.get("rl_seed") is None else int(spec["rl_seed"])
    models_dir = Path(str(spec["models_dir"]))
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
    actor_path = models_dir / f"{episode}_actor.pt"
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    critic = None
    if plot_critic_diagnostic:
        critic = model.Critic(s_dim, a_dim).to(device)
        critic_path = models_dir / f"{episode}_critic.pt"
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
    final_error = np.zeros((n_eval,), dtype=np.float32)
    integrated_error_values = np.zeros((n_eval,), dtype=np.float32)
    t_stab = np.full((n_eval,), np.nan, dtype=np.float32)
    action_curves = np.zeros((n_eval, max_steps, a_dim), dtype=np.float32)
    action_norm2_curves = np.zeros((n_eval, max_steps), dtype=np.float32)
    effort = np.zeros((n_eval,), dtype=np.float32)
    value_curves = np.zeros((n_eval, max_steps), dtype=np.float32) if plot_critic_diagnostic else None

    with torch.no_grad():
        for k in range(n_eval):
            obs = np.float32(init_states[int(init_rows[k])].copy())
            for t in range(max_steps):
                state = np.float32(obs[sensor_indices])
                st = torch.from_numpy(state).to(device).unsqueeze(0)
                at = actor(st)
                action = at.squeeze(0).cpu().numpy()
                action_curves[k, t] = action
                action_norm2_curves[k, t] = float(np.sum(np.square(action)))

                if critic is not None and value_curves is not None:
                    qt = critic(st, at)
                    value_curves[k, t] = float(qt.item())

                obs = ks.advance(obs, action)
                error_curves[k, t] = float(np.linalg.norm(obs - u_target))

            final_error[k] = float(np.mean(error_curves[k, final_start:]))
            integrated_error_values[k] = float(integrated_error(error_curves[k], dt=dt))
            effort[k] = float(integrated_control_effort(action_curves[k], dt=dt))
            hit = first_stable_step(error_curves[k], epsilon=epsilon, dwell_steps=dwell_steps)
            if hit is not None:
                t_stab[k] = np.float32(hit * dt)

    if not np.all(np.isfinite(error_curves)):
        raise ValueError(f"Non-finite error values detected for model '{name}'.")

    success_mask = ~np.isnan(t_stab)
    success_rate = float(np.mean(success_mask))

    summary = {
        "name": name,
        "role": role,
        "setup_name": setup_name,
        "target_name": target_name,
        "rl_seed": rl_seed,
        "episode": int(episode),
        "s_dim": s_dim,
        "a_dim": a_dim,
        "a_max": a_max,
        "sensor_indices": sensor_indices.tolist(),
        "dt": dt,
        "dwell_steps": int(dwell_steps),
        "metric1_control_error": {
            "mean": float(np.mean(final_error)),
            "std": _sample_std(final_error),
            "median": float(np.median(final_error)),
        },
        "metric1_integrated_error": {
            "mean": float(np.mean(integrated_error_values)),
            "std": _sample_std(integrated_error_values),
            "median": float(np.median(integrated_error_values)),
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
        },
    }

    return {
        "name": name,
        "role": role,
        "setup_name": setup_name,
        "target_name": target_name,
        "rl_seed": rl_seed,
        "episode": int(episode),
        "dt": dt,
        "sensor_indices": sensor_indices,
        "error_curves": error_curves,
        "final_error": final_error,
        "integrated_error": integrated_error_values,
        "t_stab": t_stab,
        "action_curves": action_curves,
        "action_norm2_curves": action_norm2_curves,
        "effort": effort,
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


def _require_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore
        import plotly.io as pio  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Plotly backend requested but missing dependency. "
            "Install with: pip install plotly kaleido"
        ) from exc
    return go, pio


def _save_matplotlib_with_vectors(fig: plt.Figure, out_png: Path, dpi: int = 180) -> None:
    # Force poster-quality raster export while keeping vector copies.
    dpi_hi = max(int(dpi), 600)
    fig.savefig(out_png, dpi=dpi_hi)
    fig.savefig(out_png.with_suffix(".svg"))
    fig.savefig(out_png.with_suffix(".pdf"))


def _save_plotly_with_vectors(
    fig: Any,
    out_html: Path,
    out_png: Path,
    config: Dict[str, Any],
) -> None:
    fig.write_html(out_html, include_plotlyjs=True, full_html=True, config=config)
    try:
        # High-resolution raster export for posters/slides.
        fig.write_image(out_png, scale=4)
        fig.write_image(out_png.with_suffix(".svg"))
        fig.write_image(out_png.with_suffix(".pdf"))
    except Exception as exc:
        raise RuntimeError(
            "Plotly image export failed. Ensure kaleido is installed: pip install kaleido"
        ) from exc


def _build_model_color_map(model_outputs: List[Dict[str, Any]]) -> Dict[str, str]:
    cmap = plt.get_cmap("tab10")
    out: Dict[str, str] = {}
    for i, m in enumerate(model_outputs):
        out[m["name"]] = mcolors.to_hex(cmap(i % 10))
    return out


def _pretty_model_label(name: str) -> str:
    m = MODEL_K_RE.match(str(name))
    if m:
        return f"k={int(m.group(1))}"
    return str(name)


def _error_curve_from_outputs(model_output: Dict[str, Any], curve_kind: str) -> np.ndarray:
    curves = model_output["error_curves"]
    if curve_kind == "mean":
        return np.mean(curves, axis=0)
    if curve_kind == "median":
        return np.median(curves, axis=0)
    raise ValueError(f"Unsupported curve_kind: {curve_kind}")


def _plot_error_curve_matplotlib(
    out_png: Path,
    time_axis: np.ndarray,
    model_outputs: List[Dict[str, Any]],
    color_map: Dict[str, str],
    curve_kind: str,
    title: str,
    x_label: str,
    y_label: str,
    line_width: float,
    epsilon: float,
    show_epsilon: bool,
    width_px: int,
    height_px: int,
) -> None:
    fig_w = max(1.0, width_px / 100.0)
    fig_h = max(1.0, height_px / 100.0)
    axis_title_size = 25
    y_axis_title_size = axis_title_size
    axis_tick_size = 20
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for m in model_outputs:
        curve = _error_curve_from_outputs(m, curve_kind=curve_kind)
        ax.plot(
            time_axis,
            curve,
            color=color_map[m["name"]],
            linewidth=line_width,
            label=_pretty_model_label(m["name"]),
        )
    if show_epsilon:
        ax.axhline(
            epsilon,
            color="k",
            linestyle=":",
            linewidth=max(1.0, 0.8 * line_width),
            label=f"epsilon={epsilon:.4g}",
        )
    ax.set_title(title, loc="center", pad=2)
    ax.set_xlabel(x_label, fontsize=axis_title_size, fontweight="normal")
    y_label_mpl = y_label.replace("<sub>2</sub>", "₂")
    ax.set_ylabel(y_label_mpl, fontsize=y_axis_title_size, fontweight="normal", labelpad=4)
    ax.tick_params(axis="x", labelsize=axis_tick_size, width=1.8)
    ax.tick_params(axis="y", labelsize=axis_tick_size, width=1.8)
    ax.grid(alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
        prop={"size": 18},
    )
    # Reserve right margin for legend so it never overlaps curves.
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    _save_matplotlib_with_vectors(fig, out_png, dpi=100)
    plt.close(fig)


def _plot_error_curve_plotly(
    out_png: Path,
    out_html: Path,
    time_axis: np.ndarray,
    model_outputs: List[Dict[str, Any]],
    color_map: Dict[str, str],
    curve_kind: str,
    title: str,
    x_label: str,
    y_label: str,
    line_width: float,
    epsilon: float,
    show_epsilon: bool,
    width_px: int,
    height_px: int,
    template: str,
) -> None:
    go, _ = _require_plotly()
    axis_title_size = 25
    y_axis_title_size = axis_title_size
    axis_tick_size = 20
    margin_l = 80
    margin_r = 260
    # Center title above the plotting area (exclude right legend margin).
    title_x = (1.0 + (margin_l / float(width_px)) - (margin_r / float(width_px))) / 2.0
    fig = go.Figure()
    for m in model_outputs:
        curve = _error_curve_from_outputs(m, curve_kind=curve_kind)
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=curve,
                mode="lines",
                name=_pretty_model_label(m["name"]),
                line={"width": line_width, "color": color_map[m["name"]]},
            )
        )
    if show_epsilon:
        fig.add_trace(
            go.Scatter(
                x=[float(time_axis[0]), float(time_axis[-1])],
                y=[float(epsilon), float(epsilon)],
                mode="lines",
                name=f"epsilon={epsilon:.4g}",
                line={"width": max(1.0, 0.8 * line_width), "dash": "dot", "color": "black"},
            )
        )
    fig.update_layout(
        title={
            "text": title,
            "x": title_x,
            "xanchor": "center",
            "y": 0.965,
            "yanchor": "top",
            "pad": {"t": 2, "b": 0},
            "font": {"size": 28},
        },
        xaxis={
            "title": {"text": x_label, "font": {"size": axis_title_size}, "standoff": 10},
            "tickfont": {"size": axis_tick_size},
            "showline": True,
            "linecolor": "black",
            "linewidth": 1.4,
            "mirror": True,
        },
        yaxis={
            "title": {"text": y_label, "font": {"size": y_axis_title_size}, "standoff": 4},
            "tickfont": {"size": axis_tick_size},
            "showline": True,
            "linecolor": "black",
            "linewidth": 1.4,
            "mirror": True,
        },
        template=template,
        width=width_px,
        height=height_px,
        legend={
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
            "font": {"size": 18},
            "bgcolor": "rgba(255,255,255,0.95)",
            "bordercolor": "rgba(80,80,80,1.0)",
            "borderwidth": 1,
        },
        margin={"l": margin_l, "r": margin_r, "t": 72, "b": 80},
        font={"size": 16},
    )
    config = {
        "editable": True,
        "edits": {
            "axisTitleText": True,
            "titleText": True,
            "legendPosition": True,
            "legendText": True,
        },
        "displaylogo": False,
    }
    _save_plotly_with_vectors(fig, out_html=out_html, out_png=out_png, config=config)


def _plot_error_split(
    outdir: Path,
    time_axis: np.ndarray,
    model_outputs: List[Dict[str, Any]],
    epsilon: float,
    args: argparse.Namespace,
) -> Dict[str, Optional[str]]:
    color_map = _build_model_color_map(model_outputs)
    mean_png = outdir / "error_mean_vs_time.png"
    median_png = outdir / "error_median_vs_time.png"
    mean_html: Optional[Path] = None
    median_html: Optional[Path] = None

    if args.error_plot_engine == "plotly":
        mean_html = outdir / "error_mean_vs_time.html"
        median_html = outdir / "error_median_vs_time.html"
        _plot_error_curve_plotly(
            out_png=mean_png,
            out_html=mean_html,
            time_axis=time_axis,
            model_outputs=model_outputs,
            color_map=color_map,
            curve_kind="mean",
            title=args.error_title_mean,
            x_label=args.error_x_label,
            y_label=args.error_y_label,
            line_width=args.error_line_width,
            epsilon=epsilon,
            show_epsilon=args.error_show_epsilon,
            width_px=args.error_plot_width,
            height_px=args.error_plot_height,
            template=args.error_template,
        )
        _plot_error_curve_plotly(
            out_png=median_png,
            out_html=median_html,
            time_axis=time_axis,
            model_outputs=model_outputs,
            color_map=color_map,
            curve_kind="median",
            title=args.error_title_median,
            x_label=args.error_x_label,
            y_label=args.error_y_label,
            line_width=args.error_line_width,
            epsilon=epsilon,
            show_epsilon=args.error_show_epsilon,
            width_px=args.error_plot_width,
            height_px=args.error_plot_height,
            template=args.error_template,
        )
    else:
        _plot_error_curve_matplotlib(
            out_png=mean_png,
            time_axis=time_axis,
            model_outputs=model_outputs,
            color_map=color_map,
            curve_kind="mean",
            title=args.error_title_mean,
            x_label=args.error_x_label,
            y_label=args.error_y_label,
            line_width=args.error_line_width,
            epsilon=epsilon,
            show_epsilon=args.error_show_epsilon,
            width_px=args.error_plot_width,
            height_px=args.error_plot_height,
        )
        _plot_error_curve_matplotlib(
            out_png=median_png,
            time_axis=time_axis,
            model_outputs=model_outputs,
            color_map=color_map,
            curve_kind="median",
            title=args.error_title_median,
            x_label=args.error_x_label,
            y_label=args.error_y_label,
            line_width=args.error_line_width,
            epsilon=epsilon,
            show_epsilon=args.error_show_epsilon,
            width_px=args.error_plot_width,
            height_px=args.error_plot_height,
        )

    legacy_alias = outdir / "plot_1_1_error_vs_time.png"
    shutil.copy2(mean_png, legacy_alias)
    return {
        "error_mean_png": str(mean_png),
        "error_mean_svg": str(mean_png.with_suffix(".svg")),
        "error_mean_pdf": str(mean_png.with_suffix(".pdf")),
        "error_median_png": str(median_png),
        "error_median_svg": str(median_png.with_suffix(".svg")),
        "error_median_pdf": str(median_png.with_suffix(".pdf")),
        "error_mean_html": str(mean_html) if mean_html is not None else None,
        "error_median_html": str(median_html) if median_html is not None else None,
        "legacy_mean_alias_png": str(legacy_alias),
    }


def _plot_final_error_boxplot(outdir: Path, model_outputs: List[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [m["final_error"] for m in model_outputs]
    labels = [m["name"] for m in model_outputs]
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, showfliers=False)
    _set_box_axis_limits(bp, ax, horizontal=False)
    ax.set_title("Final Control Error Distribution", loc="center")
    ax.set_ylabel(r"$\overline{\|v - E_{3}\|_{2}}_{\mathrm{final\ window}}$")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_matplotlib_with_vectors(fig, outdir / "plot_1_2_final_error_boxplot.png", dpi=180)
    plt.close(fig)


def _plot_success_rate(
    outdir: Path,
    model_outputs: List[Dict[str, Any]],
    filename: str = "plot_2_1_success_rate.png",
    title: str = "Stabilisation Success Rate",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [m["name"] for m in model_outputs]
    sr = [float(np.mean(~np.isnan(m["t_stab"]))) for m in model_outputs]
    ax.bar(np.arange(len(labels)), sr, color="0.4")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title, loc="center")
    ax.set_ylabel("Success Rate")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_matplotlib_with_vectors(fig, outdir / filename, dpi=180)
    plt.close(fig)


def _plot_tstab_success_only(
    outdir: Path,
    model_outputs: List[Dict[str, Any]],
    filename: str = "plot_2_2_tstab_boxplot_success_only.png",
    title: str = "Time-to-Stabilise (Successful Runs Only)",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    labels: List[str] = []
    data: List[np.ndarray] = []

    for m in model_outputs:
        s = m["t_stab"]
        s_ok = s[~np.isnan(s)]
        if s_ok.size == 0:
            continue
        labels.append(_pretty_model_label(m["name"]))
        data.append(s_ok)

    if data:
        bp = ax.boxplot(
            data,
            tick_labels=labels,
            showmeans=True,
            showfliers=False,
            meanprops={
                "marker": "^",
                "markerfacecolor": "tab:green",
                "markeredgecolor": "tab:green",
                "markersize": 7,
            },
            medianprops={
                "color": "tab:orange",
                "linewidth": 1.8,
            },
        )
        _set_box_axis_limits(bp, ax, horizontal=False)
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="None",
                markerfacecolor="tab:green",
                markeredgecolor="tab:green",
                markersize=7,
                label="Mean",
            ),
            Line2D(
                [0],
                [0],
                color="tab:orange",
                linewidth=1.8,
                label="Median",
            ),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=True, prop={"size": 14})
    else:
        ax.text(0.5, 0.5, "No successful stabilisations in any model.", ha="center", va="center")
        ax.set_xticks([])

    ax.set_title(title, loc="center")
    ax.set_ylabel(r"$t_{\mathrm{stab}}$ (s)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_matplotlib_with_vectors(fig, outdir / filename, dpi=180)
    plt.close(fig)


def _plot_tstab_diminishing_returns(
    outdir: Path,
    model_outputs: List[Dict[str, Any]],
    method_to_tstab_by_name: Dict[str, Dict[str, np.ndarray]],
    method_to_epsilon: Dict[str, float],
    filename: str = "plot_2_4_tstab_diminishing_returns_vs_sensor_count.png",
) -> None:
    if not method_to_tstab_by_name:
        return

    name_to_sdim = {str(m["name"]): int(m["summary"]["s_dim"]) for m in model_outputs}
    methods = list(method_to_tstab_by_name.keys())
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True, constrained_layout=True)
    ax_top, ax_bottom = axes

    for i, method in enumerate(methods):
        tstab_by_name = method_to_tstab_by_name[method]
        rows: List[Tuple[int, float, float]] = []
        for name, tstab in tstab_by_name.items():
            sdim = int(name_to_sdim[name])
            ok = tstab[~np.isnan(tstab)]
            med = float(np.nanmedian(ok)) if ok.size > 0 else np.nan
            mean = float(np.nanmean(ok)) if ok.size > 0 else np.nan
            rows.append((sdim, med, mean))
        rows.sort(key=lambda x: x[0])
        if not rows:
            continue

        xvals = np.asarray([r[0] for r in rows], dtype=np.float32)
        med_vals = np.asarray([r[1] for r in rows], dtype=np.float32)
        mean_vals = np.asarray([r[2] for r in rows], dtype=np.float32)
        c = cmap(i % 10)
        method_label = _epsilon_method_title(method)

        ax_top.plot(xvals, med_vals, color=c, linewidth=2.2, marker="o", label=f"{method_label} (median)")
        ax_top.plot(
            xvals,
            mean_vals,
            color=c,
            linewidth=1.8,
            linestyle="--",
            marker="x",
            label=f"{method_label} (mean)",
        )

        # Positive value means improvement (faster stabilization) as sensors increase.
        delta_med = np.array([np.nan] + [med_vals[j - 1] - med_vals[j] for j in range(1, med_vals.size)], dtype=np.float32)
        delta_mean = np.array([np.nan] + [mean_vals[j - 1] - mean_vals[j] for j in range(1, mean_vals.size)], dtype=np.float32)
        ax_bottom.plot(xvals, delta_med, color=c, linewidth=2.2, marker="o", label=f"{method_label} Δmedian")
        ax_bottom.plot(
            xvals,
            delta_mean,
            color=c,
            linewidth=1.8,
            linestyle="--",
            marker="x",
            label=f"{method_label} Δmean",
        )

    eps_parts = [f"{k}={v:.4g}" for k, v in method_to_epsilon.items()]
    ax_top.set_title("Time-to-Stabilise vs Sensor Count", loc="center")
    ax_top.set_ylabel(r"$t_{\mathrm{stab}}$ (s)")
    ax_top.grid(alpha=0.3)
    ax_top.legend(fontsize=13, ncol=2, frameon=True)
    if eps_parts:
        ax_top.text(
            0.99,
            0.98,
            "epsilon: " + "; ".join(eps_parts),
            transform=ax_top.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
        )

    ax_bottom.axhline(0.0, color="0.2", linewidth=1.0, linestyle=":")
    ax_bottom.set_title("Marginal Improvement with More Sensors (positive = faster)", loc="center")
    ax_bottom.set_xlabel("Number of sensors")
    ax_bottom.set_ylabel(r"$\Delta t_{\mathrm{stab}}$ (s)")
    ax_bottom.grid(alpha=0.3)
    ax_bottom.legend(fontsize=13, ncol=2, frameon=True)

    _save_matplotlib_with_vectors(fig, outdir / filename, dpi=180)
    plt.close(fig)


def _plot_diminishing_returns_suite(
    outdir: Path,
    model_outputs: List[Dict[str, Any]],
    sensor_counts: Tuple[int, ...] = (4, 8, 12, 16, 20),
    t_process: Optional[float] = None,
) -> Dict[str, Any]:
    uniform_by_k: Dict[int, Dict[str, Any]] = {}
    for m in model_outputs:
        name = str(m["name"]).lower()
        k = int(m["summary"]["s_dim"])
        if not name.startswith("uniform_k"):
            continue
        if k not in sensor_counts:
            continue
        if k in uniform_by_k:
            raise ValueError(f"Duplicate uniform model found for K={k}.")
        uniform_by_k[k] = m

    missing_k = [int(k) for k in sensor_counts if int(k) not in uniform_by_k]
    if missing_k:
        return {
            "sensor_counts": [int(v) for v in sensor_counts],
            "success_rate": [],
            "median_t_stab_success": [],
            "q25_t_stab_success": [],
            "q75_t_stab_success": [],
            "delta_t_stab": [],
            "mean_t_stab_for_marginal": [],
            "relative_improvement": [],
            "no_success_k": [],
            "t_process": t_process,
            "relative_improvement_skipped": True,
            "relative_improvement_skip_reason": f"Missing required uniform models for K={missing_k}.",
            "artifacts": {},
        }

    k_vals = np.asarray(sensor_counts, dtype=np.int64)
    success_rates = np.zeros((k_vals.size,), dtype=np.float32)
    means = np.full((k_vals.size,), np.nan, dtype=np.float32)
    medians = np.full((k_vals.size,), np.nan, dtype=np.float32)
    q25 = np.full((k_vals.size,), np.nan, dtype=np.float32)
    q75 = np.full((k_vals.size,), np.nan, dtype=np.float32)
    no_success_k: List[int] = []

    for i, k in enumerate(k_vals):
        tstab = uniform_by_k[int(k)]["t_stab"]
        ok = tstab[~np.isnan(tstab)]
        success_rates[i] = float(np.mean(~np.isnan(tstab)))
        if ok.size == 0:
            no_success_k.append(int(k))
            continue
        means[i] = float(np.nanmean(ok))
        medians[i] = float(np.nanmedian(ok))
        q25[i] = float(np.nanpercentile(ok, 25.0))
        q75[i] = float(np.nanpercentile(ok, 75.0))

    note_text = ""
    if no_success_k:
        note_text = "No-success K: " + ", ".join(str(v) for v in no_success_k)

    file_median_iqr = outdir / "plot_tstab_median_iqr_vs_sensors.png"
    # Per requirement, median trend plot starts from K=8.
    mask_k_ge_8 = k_vals >= 8
    k_vals_med = k_vals[mask_k_ge_8]
    medians_med = medians[mask_k_ge_8]
    q25_med = q25[mask_k_ge_8]
    q75_med = q75[mask_k_ge_8]
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(k_vals_med, medians_med, marker="o", linewidth=3.0, color="tab:blue", label="Median")
    ax1.fill_between(k_vals_med, q25_med, q75_med, color="tab:blue", alpha=0.22, label="IQR (25-75%)")
    if note_text:
        ax1.text(
            0.02,
            0.98,
            note_text,
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
        )
    ax1.set_title("Median Time-to-Stabilise vs Number of Sensors", loc="center")
    ax1.set_xlabel("Number of sensors (K)")
    ax1.set_ylabel(r"Median $t_{\mathrm{stab}}$ (s)")
    ax1.set_xticks(k_vals_med)
    ax1.set_xticklabels([f"{int(v)}" for v in k_vals_med])
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=True, prop={"size": 14})
    fig1.tight_layout()
    _save_matplotlib_with_vectors(fig1, file_median_iqr, dpi=200)
    plt.close(fig1)

    if t_process is None:
        # Conservative fallback only if caller does not provide a process horizon.
        t_process = float(np.max(k_vals)) * 0.0 + 1.0
    t_process = float(t_process)
    if t_process <= 0.0:
        raise ValueError(f"t_process must be positive, got {t_process}.")

    # For marginal-improvement only: if a K has no successful stabilization,
    # treat its representative stabilization time as T_process.
    means_for_marginal = means.copy()
    for i, v in enumerate(means_for_marginal):
        if not np.isfinite(v):
            means_for_marginal[i] = np.float32(t_process)

    # Marginal improvement uses transitions only (e.g., 4→8, 8→12, ...).
    delta = np.full((k_vals.size - 1,), np.nan, dtype=np.float32)
    transition_labels: List[str] = []
    transition_x = np.arange(k_vals.size - 1)
    for i in range(1, k_vals.size):
        prev_mean = means_for_marginal[i - 1]
        cur_mean = means_for_marginal[i]
        transition_labels.append(f"{int(k_vals[i - 1])}\u2192{int(k_vals[i])}")
        if np.isfinite(prev_mean) and np.isfinite(cur_mean):
            delta[i - 1] = prev_mean - cur_mean

    file_marginal = outdir / "plot_tstab_marginal_improvement_vs_sensors.png"
    fig2, ax2 = plt.subplots(figsize=(10, 7.4))
    colors = ["tab:orange"] * delta.size
    ax2.bar(transition_x, delta, width=0.7, color=colors, alpha=0.85, edgecolor="0.35", linewidth=1.2)
    note_lines: List[str] = []
    if note_text:
        note_lines.append(note_text)
    if no_success_k:
        note_lines.append(
            f"T_process fallback used for no-success K: {', '.join(str(v) for v in no_success_k)} "
            f"(T_process={t_process:.3g}s)"
        )
    if note_lines:
        ax2.text(
            0.98,
            0.98,
            "\n".join(note_lines),
            transform=ax2.transAxes,
            va="top",
            fontsize=9,
            ha="right",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
        )
    ax2.axhline(0.0, color="0.2", linewidth=1.0, linestyle=":")
    ax2.set_title("Marginal Improvement in Mean Time-to-Stabilise", loc="center", pad=14)
    ax2.set_xlabel("Sensor-count transition")
    ax2.set_ylabel(r"$\Delta t_{\mathrm{stab}}$ (s)")
    ax2.set_xticks(transition_x)
    ax2.set_xticklabels(transition_labels)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    _save_matplotlib_with_vectors(fig2, file_marginal, dpi=200)
    plt.close(fig2)

    file_relative = outdir / "plot_tstab_relative_improvement_vs_sensors.png"
    relative = np.full((k_vals.size,), np.nan, dtype=np.float32)
    relative_skipped = False
    relative_skip_reason: Optional[str] = None
    base = medians[0]
    if not np.isfinite(base):
        relative_skipped = True
        relative_skip_reason = "K=4 median t_stab is NaN; relative improvement plot skipped."
    else:
        for i in range(k_vals.size):
            if np.isfinite(medians[i]):
                relative[i] = float((base - medians[i]) / base)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(k_vals, relative, marker="o", linewidth=3.0, color="tab:green")
        if note_text:
            ax3.text(
                0.02,
                0.98,
                note_text,
                transform=ax3.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
            )
        ax3.set_title("Relative Improvement vs Number of Sensors", loc="center")
        ax3.set_xlabel("Number of sensors (K)")
        ax3.set_ylabel(r"$\frac{T(4)-T(K)}{T(4)}$")
        ax3.set_xticks(k_vals)
        ax3.set_xticklabels([f"{int(v)}" for v in k_vals])
        ax3.grid(alpha=0.3)
        fig3.tight_layout()
        _save_matplotlib_with_vectors(fig3, file_relative, dpi=200)
        plt.close(fig3)

    file_success = outdir / "plot_success_rate_vs_sensors.png"
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(k_vals, success_rates, marker="o", linewidth=3.0, color="tab:purple")
    if note_text:
        ax4.text(
            0.02,
            0.98,
            note_text,
            transform=ax4.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
        )
    ax4.set_title("Stabilisation Success Rate vs Number of Sensors", loc="center")
    ax4.set_xlabel("Number of sensors (K)")
    ax4.set_ylabel("Success rate")
    ax4.set_xticks(k_vals)
    ax4.set_xticklabels([f"{int(v)}" for v in k_vals])
    ax4.set_ylim(0.0, 1.0)
    ax4.grid(alpha=0.3)
    fig4.tight_layout()
    _save_matplotlib_with_vectors(fig4, file_success, dpi=200)
    plt.close(fig4)

    return {
        "sensor_counts": [int(v) for v in k_vals.tolist()],
        "success_rate": [float(v) for v in success_rates.tolist()],
        "median_t_stab_success": [float(v) if np.isfinite(v) else None for v in medians.tolist()],
        "mean_t_stab_success": [float(v) if np.isfinite(v) else None for v in means.tolist()],
        "q25_t_stab_success": [float(v) if np.isfinite(v) else None for v in q25.tolist()],
        "q75_t_stab_success": [float(v) if np.isfinite(v) else None for v in q75.tolist()],
        "delta_t_stab": [float(v) if np.isfinite(v) else None for v in delta.tolist()],
        "mean_t_stab_for_marginal": [float(v) if np.isfinite(v) else None for v in means_for_marginal.tolist()],
        "t_process": float(t_process),
        "relative_improvement": [float(v) if np.isfinite(v) else None for v in relative.tolist()],
        "no_success_k": [int(v) for v in no_success_k],
        "relative_improvement_skipped": bool(relative_skipped),
        "relative_improvement_skip_reason": relative_skip_reason,
        "artifacts": {
            "median_iqr_vs_sensors_png": str(file_median_iqr),
            "median_iqr_vs_sensors_svg": str(file_median_iqr.with_suffix(".svg")),
            "median_iqr_vs_sensors_pdf": str(file_median_iqr.with_suffix(".pdf")),
            "marginal_improvement_vs_sensors_png": str(file_marginal),
            "marginal_improvement_vs_sensors_svg": str(file_marginal.with_suffix(".svg")),
            "marginal_improvement_vs_sensors_pdf": str(file_marginal.with_suffix(".pdf")),
            "relative_improvement_vs_sensors_png": None if relative_skipped else str(file_relative),
            "relative_improvement_vs_sensors_svg": None if relative_skipped else str(file_relative.with_suffix(".svg")),
            "relative_improvement_vs_sensors_pdf": None if relative_skipped else str(file_relative.with_suffix(".pdf")),
            "success_rate_vs_sensors_png": str(file_success),
            "success_rate_vs_sensors_svg": str(file_success.with_suffix(".svg")),
            "success_rate_vs_sensors_pdf": str(file_success.with_suffix(".pdf")),
        },
    }


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

        ax.set_title(f"Example Stabilisation Trajectories - {m['name']}", loc="center")
        ax.set_ylabel(r"$\|v - E_{3}\|_{2}$")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=13, frameon=True)

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    _save_matplotlib_with_vectors(fig, outdir / "plot_2_3_example_stabilization_trajectories.png", dpi=180)
    plt.close(fig)


def _plot_control_trajectories_for_k(
    outdir: Path,
    time_axis: np.ndarray,
    model_outputs: List[Dict[str, Any]],
    sensor_count: int = 8,
) -> List[str]:
    if sensor_count <= 0:
        return []

    selected = [m for m in model_outputs if int(m["summary"]["s_dim"]) == int(sensor_count)]
    if not selected:
        return []

    saved: List[str] = []
    cmap = plt.get_cmap("tab10")
    for m in selected:
        actions = m["action_curves"]
        mean_curve = np.mean(actions, axis=0)
        median_curve = np.median(actions, axis=0)
        a_dim = int(mean_curve.shape[1])

        fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
        for j in range(a_dim):
            c = cmap(j % 10)
            axes[0].plot(time_axis, mean_curve[:, j], color=c, linewidth=2.0, label=f"actuator {j+1}")
            axes[1].plot(time_axis, median_curve[:, j], color=c, linewidth=2.0, label=f"actuator {j+1}")

        axes[0].set_title(f"Mean Control Trajectories vs Time - {m['name']}", loc="center")
        axes[0].set_ylabel(r"$u_j(t)$")
        axes[0].grid(alpha=0.3)
        axes[0].legend(ncol=2, fontsize=13, frameon=True)

        axes[1].set_title(f"Median Control Trajectories vs Time - {m['name']}", loc="center")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel(r"$u_j(t)$")
        axes[1].grid(alpha=0.3)
        axes[1].legend(ncol=2, fontsize=13, frameon=True)

        fig.tight_layout()
        out_png = outdir / f"plot_control_trajectories_{sanitize_key(m['name'])}.png"
        _save_matplotlib_with_vectors(fig, out_png, dpi=200)
        plt.close(fig)
        saved.append(str(out_png))

    if sensor_count == 8 and len(saved) == 1:
        alias = outdir / "plot_control_trajectories_k8.png"
        shutil.copy2(saved[0], alias)
        saved.append(str(alias))

    return saved


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
    ax.set_title("Control Effort vs Final Error", loc="center")
    ax.set_xlabel("J_u")
    ax.set_ylabel(r"$\overline{\|v - E_{3}\|_{2}}_{\mathrm{final\ window}}$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13, frameon=True)
    fig.tight_layout()
    _save_matplotlib_with_vectors(fig, outdir / "plot_3_1_effort_vs_final_error.png", dpi=180)
    plt.close(fig)


def _plot_effort_boxplot(outdir: Path, model_outputs: List[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [m["effort"] for m in model_outputs]
    labels = [m["name"] for m in model_outputs]
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, showfliers=False)
    _set_box_axis_limits(bp, ax, horizontal=False)
    ax.set_title("Control Effort Distribution", loc="center")
    ax.set_ylabel("J_u")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_matplotlib_with_vectors(fig, outdir / "plot_3_2_effort_boxplot.png", dpi=180)
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

    ax.set_title("Mean Control Magnitude vs Time", loc="center")
    ax.set_xlabel("Time")
    ax.set_ylabel("||u(t)||_2")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13, frameon=True)
    fig.tight_layout()
    _save_matplotlib_with_vectors(fig, outdir / "plot_3_3_control_magnitude_vs_time.png", dpi=180)
    plt.close(fig)


def _plot_paired_final_error_diff(outdir: Path, model_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline = next((m for m in model_outputs if m["role"] == "baseline"), None)
    candidate = next((m for m in model_outputs if m["role"] == "candidate"), None)
    if baseline is None or candidate is None:
        comparable = [m for m in model_outputs if m["role"] != "reference"]
        if len(comparable) >= 2:
            baseline = comparable[0]
            candidate = comparable[1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    if baseline is None or candidate is None:
        msg = "Paired-diff unavailable: need at least two comparable models."
        axes[0].text(0.5, 0.5, msg, ha="center", va="center")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].set_visible(False)
        _save_matplotlib_with_vectors(fig, outdir / "plot_final_error_paired_diff.png", dpi=180)
        plt.close(fig)
        return {"available": False, "message": msg}

    if baseline["final_error"].shape[0] != candidate["final_error"].shape[0]:
        raise ValueError("Paired-diff requires the same number of rollouts for both comparison models.")

    delta = candidate["final_error"] - baseline["final_error"]
    frac_better = float(np.mean(delta < 0.0))
    baseline_name = str(baseline["name"])
    candidate_name = str(candidate["name"])
    delta_label = f"{candidate_name} - {baseline_name}"

    axes[0].hist(delta, bins=20, color="0.7", edgecolor="0.3")
    axes[0].axvline(0.0, color="k", linestyle=":", linewidth=1.6)
    axes[0].axvline(float(np.mean(delta)), color="tab:red", linewidth=1.8, label="mean(Delta)")
    axes[0].set_title(f"Paired Final Error Difference: Delta = {delta_label}", loc="center")
    axes[0].set_xlabel("Delta")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=13, frameon=True)
    axes[0].text(
        0.98,
        0.95,
        f"P(Delta < 0) = {frac_better:.3f}",
        ha="right",
        va="top",
        transform=axes[0].transAxes,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )

    bp = axes[1].boxplot([delta], vert=False, tick_labels=[delta_label], showmeans=True, showfliers=False)
    _set_box_axis_limits(bp, axes[1], horizontal=True)
    axes[1].axvline(0.0, color="k", linestyle=":", linewidth=1.6)
    axes[1].set_xlabel("Delta")
    axes[1].grid(alpha=0.3)

    _save_matplotlib_with_vectors(fig, outdir / "plot_final_error_paired_diff.png", dpi=180)
    plt.close(fig)
    return {
        "available": True,
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
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
        ax.set_title("Critic Value Function Evolution (Diagnostic Only)", loc="center")
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=13, frameon=True)
    fig.tight_layout()
    _save_matplotlib_with_vectors(fig, outdir / "plot_diag_value_vs_time.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    target_name = Path(args.target_file).stem
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
    if args.epsilon_mode != "target_relative":
        raise ValueError(
            "This focused evaluation workflow requires target-relative epsilon. "
            "Use --epsilon-mode target_relative."
        )
    if Path(args.init_file).name != "INIT.dat":
        raise ValueError(f"init-file must point to INIT.dat under the study protocol, got '{Path(args.init_file).name}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model_specs = load_models_spec(args.models_spec)
    if not args.include_reference:
        model_specs = [m for m in model_specs if str(m["role"]) not in {"full64", "reference"}]
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

    split_manifest = load_or_create_split_manifest(
        path=Path(args.init_split_file),
        init_file=args.init_file,
        num_rows=int(init_states.shape[0]),
        split_seed=int(args.split_seed),
        train_size=int(args.train_split_size),
        val_size=int(args.val_split_size),
        test_size=int(args.test_split_size),
    )

    final_start = _final_window_start(
        max_steps=args.max_steps,
        final_window_frac=args.final_window_frac,
        final_window_steps=args.final_window_steps,
    )

    role_key = f"{args.split_role}_rows"
    init_rows = np.asarray(split_manifest[role_key], dtype=np.int64)
    init_rows_source = "shared_split_manifest"

    baseline_mean_error: Optional[float] = None
    baseline_median_error: Optional[float] = None
    baseline_error_curves: Optional[np.ndarray] = None

    baseline_a_dim = int(model_specs[0]["a_dim"])
    baseline_error_curves, baseline_dt = _compute_uncontrolled_baseline_error_curves(
        x=x,
        u_target=u_target,
        init_states=init_states,
        init_rows=init_rows,
        max_steps=args.max_steps,
        domain_length=args.domain_length,
        a_dim=baseline_a_dim,
    )
    if abs(float(baseline_dt) - float(args.dt_expected)) > 1e-12:
        raise ValueError(
            f"Baseline dt mismatch: KS dt={baseline_dt} but dt_expected={args.dt_expected}"
        )
    baseline_mean_error = float(np.mean(baseline_error_curves))
    baseline_median_error = float(np.median(baseline_error_curves))

    epsilon_methods: Dict[str, float] = {
        "target_relative": compute_target_relative_epsilon(u_target=u_target, epsilon_beta=args.epsilon_beta),
    }
    primary_method = "target_relative"
    epsilon_source = "target_relative_target_norm"
    epsilon = float(epsilon_methods[primary_method])

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

    # Recompute stabilization metrics for all requested epsilon methods from raw error curves.
    method_to_tstab_by_name: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in epsilon_methods.keys()}
    for o in outputs:
        method_metrics: Dict[str, Dict[str, Any]] = {}
        for method, eps_val in epsilon_methods.items():
            metrics = _compute_stabilization_from_error_curves(
                error_curves=o["error_curves"],
                epsilon=float(eps_val),
                dwell_steps=int(o["summary"]["dwell_steps"]),
                dt=float(o["dt"]),
            )
            method_metrics[method] = metrics
            method_to_tstab_by_name[method][str(o["name"])] = metrics["t_stab"]
        o["stabilization_metrics_by_method"] = method_metrics
        o["t_stab"] = method_metrics[primary_method]["t_stab"]
        o["summary"]["metric2_stabilization"] = method_metrics[primary_method]["summary"]
        if len(epsilon_methods) > 1:
            o["summary"]["metric2_stabilization_by_epsilon"] = {
                m: method_metrics[m]["summary"] for m in epsilon_methods.keys()
            }

    dt_used = float(outputs[0]["dt"])
    time_axis = np.arange(args.max_steps, dtype=np.float32) * dt_used
    outdir = _ensure_outdir(args.outdir)
    split_group_label = "test" if args.split_role == "test" else args.split_role

    error_plot_files = _plot_error_split(outdir, time_axis, outputs, epsilon, args)
    _plot_tstab_success_only(outdir, outputs)
    _plot_effort_vs_error(outdir, outputs)
    _plot_effort_boxplot(outdir, outputs)
    _plot_control_magnitude_vs_time(outdir, time_axis, outputs)
    diminishing_meta = _plot_diminishing_returns_suite(
        outdir=outdir,
        model_outputs=outputs,
        sensor_counts=(4, 8, 12, 16, 20),
        t_process=float(args.max_steps) * float(dt_used),
    )

    per_row_records: List[Dict[str, Any]] = []
    run_summaries: List[Dict[str, Any]] = []
    grouped_runs: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for o in outputs:
        row_metrics: List[Dict[str, Any]] = []
        for idx, row_id in enumerate(init_rows.tolist()):
            success = bool(not np.isnan(o["t_stab"][idx]))
            time_to_stabilize = None if not success else float(o["t_stab"][idx])
            record = {
                "setup_name": str(o.get("setup_name", o["name"])),
                "target_name": str(o.get("target_name", target_name)),
                "rl_seed": None if o.get("rl_seed") is None else int(o["rl_seed"]),
                "model_name": str(o["name"]),
                "role": str(o["role"]),
                "split_role": args.split_role,
                "init_row": int(row_id),
                "success": success,
                "final_error": float(o["final_error"][idx]),
                "integrated_error": float(o["integrated_error"][idx]),
                "time_to_stabilize": time_to_stabilize,
                "control_effort": float(o["effort"][idx]),
            }
            per_row_records.append(record)
            row_metrics.append(record)

        run_summary = summarize_run_rows(row_metrics)
        run_summary.update(
            {
                "setup_name": str(o.get("setup_name", o["name"])),
                "target_name": str(o.get("target_name", target_name)),
                "rl_seed": None if o.get("rl_seed") is None else int(o["rl_seed"]),
                "model_name": str(o["name"]),
                "role": str(o["role"]),
                "split_role": args.split_role,
                "episode": int(o["episode"]),
            }
        )
        run_summaries.append(run_summary)
        grouped_runs.setdefault((run_summary["setup_name"], run_summary["target_name"]), []).append(run_summary)

    setup_summaries: List[Dict[str, Any]] = []
    for (setup_name, target_group), runs in grouped_runs.items():
        stats = summarize_setup_runs(runs)
        setup_record: Dict[str, Any] = {
            "setup_name": setup_name,
            "target_name": target_group,
            "num_runs": int(len(runs)),
            "split_role": args.split_role,
        }
        for metric_name, metric_stats in stats.items():
            setup_record[f"{metric_name}_mean"] = float(metric_stats["mean"])
            setup_record[f"{metric_name}_std"] = float(metric_stats["std"])
            setup_record[f"{metric_name}_median"] = float(metric_stats["median"])
        setup_summaries.append(setup_record)

    setup_summaries.sort(
        key=lambda row: (
            -float(row["success_rate_mean"]),
            float(row["mean_final_error_mean"]),
            float(row["mean_control_effort_mean"]),
            float(row["success_rate_std"]),
        )
    )
    for rank, row in enumerate(setup_summaries, start=1):
        row["rank"] = int(rank)

    run_config = {
        "models_spec_path": args.models_spec,
        "models_evaluated": [
            {
                "name": o["name"],
                "role": o["role"],
                "setup_name": o.get("setup_name", o["name"]),
                "target_name": o.get("target_name", target_name),
                "rl_seed": o.get("rl_seed", None),
                "episode": o["episode"],
                "sensor_indices": o["sensor_indices"].tolist(),
            }
            for o in outputs
        ],
        "state_dim": args.state_dim,
        "num_evals": int(init_rows.size),
        "max_steps": args.max_steps,
        "domain_length": args.domain_length,
        "target_file": args.target_file,
        "x_file": args.x_file,
        "init_file": args.init_file,
        "init_split_file": args.init_split_file,
        "split_seed": int(args.split_seed),
        "train_split_size": int(args.train_split_size),
        "val_split_size": int(args.val_split_size),
        "test_split_size": int(args.test_split_size),
        "split_role": args.split_role,
        "seed": args.seed,
        "init_rows": init_rows.tolist(),
        "init_rows_source": init_rows_source,
        "dt_expected": args.dt_expected,
        "dt_used": dt_used,
        "epsilon_method": epsilon_source,
        "epsilon_mode": args.epsilon_mode,
        "epsilon_primary_method": primary_method,
        "epsilon_methods": {k: float(v) for k, v in epsilon_methods.items()},
        "epsilon_beta": args.epsilon_beta,
        "epsilon_value": epsilon,
        "baseline_mean_error": baseline_mean_error,
        "baseline_median_error": baseline_median_error,
        "error_norm": "l2_full_state",
        "dwell_time": args.dwell_time,
        "dwell_steps": int(np.ceil(args.dwell_time / dt_used)),
        "final_window_start_idx": int(final_start),
        "final_window_steps": int(args.max_steps - final_start),
        "include_reference": bool(args.include_reference),
        "plot_critic_diagnostic": bool(args.plot_critic_diagnostic),
        "error_plot_config": {
            "engine": args.error_plot_engine,
            "show_epsilon": bool(args.error_show_epsilon),
            "width_px": int(args.error_plot_width),
            "height_px": int(args.error_plot_height),
            "line_width": float(args.error_line_width),
            "template": args.error_template,
            "x_label": args.error_x_label,
            "y_label": args.error_y_label,
            "title_mean": args.error_title_mean,
            "title_median": args.error_title_median,
            "artifacts": error_plot_files,
        },
        "plot_scope": "core_diminishing",
        "epsilon_mode_used_for_stabilization": "target_relative",
        "diminishing_sensor_counts": diminishing_meta["sensor_counts"],
        "diminishing_artifacts": diminishing_meta["artifacts"],
        "diminishing_summary": {
            "no_success_k": diminishing_meta["no_success_k"],
            "t_process": diminishing_meta["t_process"],
            "relative_improvement_skipped": diminishing_meta["relative_improvement_skipped"],
            "relative_improvement_skip_reason": diminishing_meta["relative_improvement_skip_reason"],
        },
        "comparison_protocol": {
            "checkpoint_rule": [
                "highest validation success rate",
                "tie-break by lower mean final error",
                "tie-break by lower control effort",
            ],
            "setup_ranking_rule": [
                "highest mean test success rate",
                "then lowest mean test final tracking error",
                "then lowest mean control effort",
                "prefer lower seed-to-seed variance when similar",
            ],
        },
    }
    (outdir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    summary = {
        "epsilon": epsilon,
        "epsilon_mode": args.epsilon_mode,
        "epsilon_primary_method": primary_method,
        "epsilon_methods": {k: float(v) for k, v in epsilon_methods.items()},
        "baseline_mean_error": baseline_mean_error,
        "baseline_median_error": baseline_median_error,
        "dt": dt_used,
        "models": {o["name"]: o["summary"] for o in outputs},
        "split_role": args.split_role,
        "per_run": run_summaries,
        "per_setup": setup_summaries,
        "diminishing_returns": {
            "sensor_counts": diminishing_meta["sensor_counts"],
            "success_rate": diminishing_meta["success_rate"],
            "median_t_stab_success": diminishing_meta["median_t_stab_success"],
            "q25_t_stab_success": diminishing_meta["q25_t_stab_success"],
            "q75_t_stab_success": diminishing_meta["q75_t_stab_success"],
            "delta_t_stab": diminishing_meta["delta_t_stab"],
            "relative_improvement": diminishing_meta["relative_improvement"],
            "no_success_k": diminishing_meta["no_success_k"],
            "relative_improvement_skipped": diminishing_meta["relative_improvement_skipped"],
            "relative_improvement_skip_reason": diminishing_meta["relative_improvement_skip_reason"],
        },
    }
    (outdir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / f"{split_group_label}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    per_row_csv = outdir / f"{split_group_label}_per_row.csv"
    with per_row_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_row_records[0].keys()))
        writer.writeheader()
        writer.writerows(per_row_records)

    run_summary_csv = outdir / f"{split_group_label}_run_summary.csv"
    with run_summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(run_summaries)

    setup_summary_csv = outdir / f"{split_group_label}_setup_summary.csv"
    with setup_summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(setup_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(setup_summaries)

    arrays: Dict[str, Any] = {
        "init_rows": init_rows,
        "time_axis": time_axis,
        "epsilon": np.float32(epsilon),
        "dt": np.float32(dt_used),
    }
    for method, eps_val in epsilon_methods.items():
        arrays[f"epsilon_{method}"] = np.float32(eps_val)
    if baseline_error_curves is not None:
        arrays["baseline_error_curves"] = baseline_error_curves
    for i, o in enumerate(outputs):
        key = f"{i}_{sanitize_key(o['name'])}"
        arrays[f"{key}__role"] = np.asarray([o["role"]], dtype=object)
        arrays[f"{key}__error_curves"] = o["error_curves"]
        arrays[f"{key}__final_error"] = o["final_error"]
        arrays[f"{key}__integrated_error"] = o["integrated_error"]
        arrays[f"{key}__t_stab"] = o["t_stab"]
        arrays[f"{key}__action_curves"] = o["action_curves"]
        arrays[f"{key}__action_norm2_curves"] = o["action_norm2_curves"]
        arrays[f"{key}__effort"] = o["effort"]
        if "stabilization_metrics_by_method" in o:
            for method, metrics in o["stabilization_metrics_by_method"].items():
                arrays[f"{key}__t_stab_{method}"] = metrics["t_stab"]
        arrays[f"{key}__sensor_indices"] = o["sensor_indices"]
        arrays[f"{key}__episode"] = np.int64(o["episode"])
        if o["value_curves"] is not None:
            arrays[f"{key}__value_curves"] = o["value_curves"]

    np.savez_compressed(outdir / "metrics_arrays.npz", **arrays)

    print(f"Saved run config: {outdir / 'run_config.json'}")
    print(f"Saved summary metrics: {outdir / 'metrics_summary.json'}")
    print(f"Saved protocol summary: {outdir / f'{split_group_label}_summary.json'}")
    print(f"Saved per-row metrics: {per_row_csv}")
    print(f"Saved run summaries: {run_summary_csv}")
    print(f"Saved setup summaries: {setup_summary_csv}")
    print(f"Saved arrays: {outdir / 'metrics_arrays.npz'}")
    print(f"Saved plots in: {outdir}")
    print(f"Epsilon (target-relative): {epsilon:.6f} = {args.epsilon_beta:.3f} * ||E3||_2")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
