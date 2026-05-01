from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ks_control.ks import KS


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot sensor and actuator positions on the spatial grid from x.dat."
    )
    ap.add_argument(
        "--sensor-indices",
        type=str,
        required=True,
        help="Sensor indices as comma list/ranges (e.g. '0,4,8-11') or path to a JSON list file.",
    )
    ap.add_argument("--x-file", type=str, default=str(DATA_DIR / "x.dat"))
    ap.add_argument("--a-dim", type=int, default=4)
    ap.add_argument("--a-max", type=float, default=1.0, help="Action amplitude limit used when --action-values is not provided.")
    ap.add_argument(
        "--action-values",
        type=str,
        default=None,
        help="Optional comma-separated actuator actions (length a-dim) to plot actual applied force B[:,i]*u_i.",
    )
    ap.add_argument("--domain-length", type=float, default=22.0)
    overlay_group = ap.add_mutually_exclusive_group()
    overlay_group.add_argument(
        "--overlay-forcing",
        dest="overlay_forcing",
        action="store_true",
        help="Overlay actuator forcing distributions B[:, i] on a secondary y-axis.",
    )
    overlay_group.add_argument(
        "--no-overlay-forcing",
        dest="overlay_forcing",
        action="store_false",
        help="Disable forcing-distribution overlay.",
    )
    ap.set_defaults(overlay_forcing=True)
    ap.add_argument("--forcing-linewidth", type=float, default=1.6)
    ap.add_argument("--forcing-alpha", type=float, default=0.85)
    ap.add_argument(
        "--actuator-marker-size",
        type=float,
        default=260.0,
        help="Triangle marker area for actuator symbols.",
    )
    ap.add_argument(
        "--legend-sensor-marker-size",
        type=float,
        default=10.0,
        help="Legend marker size for sensors (points).",
    )
    ap.add_argument(
        "--legend-actuator-marker-size",
        type=float,
        default=10.0,
        help="Legend marker size for actuators (points).",
    )
    ap.add_argument(
        "--legend-font-size",
        type=float,
        default=14.0,
        help="Legend font size.",
    )
    ap.add_argument(
        "--title-font-size",
        type=float,
        default=20.0,
        help="Plot title font size.",
    )
    ap.add_argument(
        "--legend-cols",
        type=int,
        default=4,
        help="Number of legend columns.",
    )
    ap.add_argument(
        "--output-plot",
        type=str,
        default=str(ROOT / "studies" / "sensor_placement" / "results" / "sensor_actuator_positions.png"),
    )
    ap.add_argument(
        "--output-json",
        type=str,
        default=str(ROOT / "studies" / "sensor_placement" / "results" / "sensor_actuator_positions.json"),
    )
    ap.add_argument("--no-show", action="store_true")
    return ap.parse_args()


def _parse_index_expr(expr: str) -> List[int]:
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
        raise ValueError("No sensor indices parsed.")
    return out


def parse_sensor_indices(arg: str) -> List[int]:
    p = Path(arg)
    if p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "sensor_indices" in data:
                data = data["sensor_indices"]
            elif "indices" in data:
                data = data["indices"]
            else:
                raise ValueError("JSON object must contain key 'sensor_indices' or 'indices'.")
        if not isinstance(data, list):
            raise ValueError("Sensor index JSON must be a list.")
        out: List[int] = []
        seen = set()
        for item in data:
            idx = int(str(item).replace("s", "")) if isinstance(item, str) else int(item)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
        if not out:
            raise ValueError("Sensor index JSON is empty.")
        return out
    return _parse_index_expr(arg)


def parse_action_values(expr: str, a_dim: int) -> np.ndarray:
    vals = [float(tok.strip()) for tok in expr.split(",") if tok.strip()]
    if len(vals) != a_dim:
        raise ValueError(f"--action-values must contain exactly {a_dim} values, got {len(vals)}.")
    return np.asarray(vals, dtype=np.float32)


def main() -> None:
    args = parse_args()
    if args.a_dim <= 0:
        raise ValueError("a-dim must be positive.")
    if args.domain_length <= 0:
        raise ValueError("domain-length must be positive.")
    if args.a_max <= 0:
        raise ValueError("a-max must be positive.")
    if args.forcing_linewidth <= 0:
        raise ValueError("forcing-linewidth must be positive.")
    if args.forcing_alpha < 0.0 or args.forcing_alpha > 1.0:
        raise ValueError("forcing-alpha must be in [0, 1].")
    if args.actuator_marker_size <= 0:
        raise ValueError("actuator-marker-size must be positive.")
    if args.legend_sensor_marker_size <= 0:
        raise ValueError("legend-sensor-marker-size must be positive.")
    if args.legend_actuator_marker_size <= 0:
        raise ValueError("legend-actuator-marker-size must be positive.")
    if args.legend_font_size <= 0:
        raise ValueError("legend-font-size must be positive.")
    if args.legend_cols <= 0:
        raise ValueError("legend-cols must be positive.")
    if args.title_font_size <= 0:
        raise ValueError("title-font-size must be positive.")

    x = np.loadtxt(args.x_file)
    if x.ndim != 1:
        raise ValueError(f"x-file must be 1D, got shape {x.shape}")

    sensor_indices = parse_sensor_indices(args.sensor_indices)
    bad = [i for i in sensor_indices if i < 0 or i >= x.size]
    if bad:
        raise ValueError(f"Sensor index out of bounds for x size {x.size}: {bad}")
    if args.action_values is not None:
        action_values = parse_action_values(args.action_values, args.a_dim)
    else:
        action_values = np.full((args.a_dim,), args.a_max, dtype=np.float32)

    ks = KS(L=args.domain_length, N=x.size, a_dim=args.a_dim)
    actuator_indices = [int(np.argmax(ks.B[:, i])) for i in range(args.a_dim)]
    actuator_positions = [float(x[i]) for i in actuator_indices]
    sensor_positions = [float(x[i]) for i in sensor_indices]
    forcing_peak_indices = actuator_indices.copy()
    forcing_peak_positions = actuator_positions.copy()

    fig, ax = plt.subplots(figsize=(12, 3.8))
    # Keep the baseline line visually, but do not include it in the legend.
    ax.plot(x, np.zeros_like(x), color="0.75", linewidth=1.0, label="_nolegend_")

    sensor_y = 0.23
    sensor_label_y = 0.218
    ax.scatter(sensor_positions, np.full(len(sensor_positions), sensor_y), s=85, marker="o", color="tab:blue", label="Sensors")
    for k, idx in enumerate(sensor_indices):
        ax.text(sensor_positions[k], sensor_label_y, f"s{idx}", color="tab:blue", ha="center", va="top", fontsize=8)

    # Always draw actuator locations on the primary axis using triangles for clear placement reference.
    actuator_y = 0.0
    ax.scatter(
        actuator_positions,
        np.full(len(actuator_positions), actuator_y),
        s=args.actuator_marker_size,
        marker="^",
        color="tab:red",
        label="Actuators",
        zorder=5,
    )

    ax2 = None
    force_line_handle = None
    force_peak_handle = None
    if args.overlay_forcing:
        ax2 = ax.twinx()
        cmap = plt.get_cmap("tab10")
        x_dense = np.linspace(float(np.min(x)), float(np.max(x)), max(512, x.size * 16))
        min_applied = 0.0
        for i in range(args.a_dim):
            c = cmap(i % 10)
            applied_force = ks.B[:, i] * float(action_values[i])
            peak_idx = int(np.argmax(np.abs(applied_force)))
            min_applied = min(min_applied, float(np.min(applied_force)))

            # Smooth-looking force distribution curve via dense interpolation.
            forcing_dense = np.interp(x_dense, x, applied_force)
            # Remove near-zero tail/flat sections from display.
            cutoff = 0.01 * float(np.max(np.abs(forcing_dense)))
            forcing_dense = np.where(np.abs(forcing_dense) > cutoff, forcing_dense, np.nan)
            ax2.plot(
                x_dense,
                forcing_dense,
                color=c,
                linewidth=args.forcing_linewidth,
                alpha=args.forcing_alpha,
                antialiased=True,
                solid_capstyle="round",
            )
            ax2.scatter(
                float(x[peak_idx]),
                float(applied_force[peak_idx]),
                color=c,
                marker="x",
                s=70,
                linewidths=args.forcing_linewidth,
                alpha=args.forcing_alpha,
            )
        force_line_handle = Line2D(
            [0],
            [0],
            color="0.2",
            linewidth=args.forcing_linewidth,
            alpha=args.forcing_alpha,
            label="Applied force distribution",
        )
        force_peak_handle = Line2D(
            [0],
            [0],
            color="0.2",
            marker="x",
            linestyle="None",
            markersize=7,
            markeredgewidth=args.forcing_linewidth,
            alpha=args.forcing_alpha,
            label="Applied force peak",
        )
        ax2.set_ylabel("Applied force amplitude")
        ax2.set_ylim(bottom=min_applied)
        ax2.grid(False)
    ax.set_title("Sensor and Actuator Positions on x-grid", fontsize=float(args.title_font_size))
    ax.set_xlabel("x")
    ax.set_yticks([])
    # Show only the upper half (above y=0) for cleaner sensor-focused view.
    ax.set_ylim(0.0, 0.25)
    ax.grid(axis="x", alpha=0.25)
    sensor_legend_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="tab:blue",
        markeredgecolor="tab:blue",
        markersize=float(args.legend_sensor_marker_size),
        label="Sensors",
    )
    actuator_legend_handle = Line2D(
        [0],
        [0],
        marker="^",
        linestyle="None",
        markerfacecolor="tab:red",
        markeredgecolor="tab:red",
        markersize=float(args.legend_actuator_marker_size),
        label="Actuators",
    )
    handles1 = [sensor_legend_handle, actuator_legend_handle]
    labels1 = ["Sensors", "Actuators"]
    handles2 = []
    labels2: List[str] = []
    if force_line_handle is not None and force_peak_handle is not None:
        handles2 = [force_line_handle, force_peak_handle]
        labels2 = ["Applied force distribution", "Applied force peak"]
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.40),
        fontsize=float(args.legend_font_size),
        ncol=int(args.legend_cols),
        borderaxespad=0.0,
        frameon=True,
    )
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    fig.subplots_adjust(bottom=0.52)

    output_plot = Path(args.output_plot)
    if output_plot.parent != Path("."):
        output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=180, bbox_inches="tight", pad_inches=0.20)
    extra_plot_paths: List[Path] = []
    if output_plot.suffix.lower() == ".png":
        output_svg = output_plot.with_suffix(".svg")
        output_pdf = output_plot.with_suffix(".pdf")
        fig.savefig(output_svg, bbox_inches="tight", pad_inches=0.20)
        fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.20)
        extra_plot_paths.extend([output_svg, output_pdf])
    if not args.no_show:
        plt.show()
    plt.close(fig)

    summary = {
        "x_file": args.x_file,
        "grid_size": int(x.size),
        "domain_length": float(args.domain_length),
        "a_dim": int(args.a_dim),
        "a_max": float(args.a_max),
        "action_values_used": [float(v) for v in action_values.tolist()],
        "forcing_overlay_enabled": bool(args.overlay_forcing),
        "forcing_peak_indices": forcing_peak_indices,
        "forcing_peak_positions": forcing_peak_positions,
        "sensor_indices": sensor_indices,
        "sensor_positions": sensor_positions,
        "actuator_indices": actuator_indices,
        "actuator_positions": actuator_positions,
    }
    output_json = Path(args.output_json)
    if output_json.parent != Path("."):
        output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved plot: {output_plot}")
    for p in extra_plot_paths:
        print(f"Saved plot: {p}")
    print(f"Saved summary: {output_json}")


if __name__ == "__main__":
    main()
