from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
SENSOR_COUNT_RESULTS_DIR = ROOT / "studies" / "sensor_count" / "results"
DEFAULT_MODELS_SPEC = ROOT / "artifacts" / "raw" / "sensor_placement" / "training_bundle" / "models_spec_for_evaluation.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ks_control.ks import KS
from ks_control import model
from ks_control.evaluation_helpers import resolve_sensor_indices

plt.rcParams.update(
    {
        "font.size": 16,
        "font.weight": "normal",
        "axes.labelsize": 20,
        "axes.labelweight": "normal",
        "axes.titlesize": 22,
        "axes.titleweight": "normal",
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 1.6,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
    }
)

MODEL_K_RE = re.compile(r".*_k(\d+)$")


def _pretty_rollout_title(name: str, ckpt_label: str) -> str:
    m = MODEL_K_RE.match(str(name))
    if m:
        return f"k={int(m.group(1))}"
    return f"{name}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate three separate physical rollout heatmaps: uncontrolled, uniform_k4, and uniform_k8 "
            "using one shared initial condition and one shared color scale."
        )
    )
    ap.add_argument(
        "--models-spec",
        type=str,
        default=str(DEFAULT_MODELS_SPEC),
        help="Path to models spec JSON containing uniform_k4 and uniform_k8 entries.",
    )
    ap.add_argument("--model-k4-name", type=str, default="uniform_k4")
    ap.add_argument("--model-k8-name", type=str, default="uniform_k8")
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--domain-length", type=float, default=22.0)
    ap.add_argument("--x-file", type=str, default=str(DATA_DIR / "x.dat"))
    ap.add_argument("--init-file", type=str, default=str(DATA_DIR / "INIT.dat"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-row", type=int, default=None, help="Optional fixed INIT row.")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--cmap", type=str, default="viridis")
    ap.add_argument(
        "--interpolation",
        type=str,
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
    )
    ap.add_argument("--dpi", type=int, default=600, help="High-resolution PNG DPI.")
    ap.add_argument("--q-low", type=float, default=1.0, help="Lower quantile for shared color scale.")
    ap.add_argument("--q-high", type=float, default=99.0, help="Upper quantile for shared color scale.")
    ap.add_argument("--title-font-size", type=float, default=30.0, help="Heatmap title font size.")
    ap.add_argument("--axis-label-font-size", type=float, default=26.0, help="Axis label font size.")
    ap.add_argument("--tick-font-size", type=float, default=22.0, help="Axis tick label font size.")
    ap.add_argument("--colorbar-label-font-size", type=float, default=24.0, help="Colorbar label font size.")
    ap.add_argument("--colorbar-tick-font-size", type=float, default=20.0, help="Colorbar tick font size.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(SENSOR_COUNT_RESULTS_DIR / "rollout_triplet"),
    )
    return ap.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models_spec(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"models-spec file not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("models", None)
    if not isinstance(data, list) or not data:
        raise ValueError("models-spec must be a non-empty list or {\"models\": [...]} object.")
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(data):
        if not isinstance(m, dict):
            raise ValueError(f"models-spec entry #{i} must be an object.")
        for req in ["name", "models_dir", "s_dim", "a_dim", "a_max"]:
            if req not in m:
                raise ValueError(f"models-spec entry #{i} missing required field '{req}'.")
        out.append(m)
    return out


def pick_actor_checkpoint(spec: Dict[str, Any]) -> Tuple[Path, str]:
    models_dir = Path(str(spec["models_dir"]))
    req_ep = spec.get("episode", None)
    if req_ep is not None:
        p = models_dir / f"{int(req_ep)}_actor.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing actor checkpoint: {p}")
        return p, str(int(req_ep))

    best = models_dir / "best_actor.pt"
    if best.exists():
        return best, "best"

    actor_ckpts = sorted(models_dir.glob("*_actor.pt"))
    numeric_eps: List[int] = []
    for p in actor_ckpts:
        stem = p.name.split("_")[0]
        if stem.isdigit():
            numeric_eps.append(int(stem))
    if not numeric_eps:
        raise FileNotFoundError(f"No actor checkpoint found in {models_dir}")
    ep = max(numeric_eps)
    return models_dir / f"{ep}_actor.pt", str(ep)


def rollout_controlled(
    spec: Dict[str, Any],
    init_state: np.ndarray,
    state_dim: int,
    domain_length: float,
    max_steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    name = str(spec["name"])
    s_dim = int(spec["s_dim"])
    a_dim = int(spec["a_dim"])
    a_max = float(spec["a_max"])
    sensor_idx = resolve_sensor_indices(spec, state_dim)
    actor_ckpt, ckpt_label = pick_actor_checkpoint(spec)

    actor = model.Actor(s_dim, a_dim, a_max).to(device)
    actor.load_state_dict(torch.load(actor_ckpt, map_location=device))
    actor.eval()

    ks = KS(L=domain_length, N=state_dim, a_dim=a_dim)
    u = np.float32(init_state.copy())
    hist = np.zeros((max_steps + 1, state_dim), dtype=np.float32)
    hist[0] = u

    with torch.no_grad():
        for t in range(max_steps):
            s = np.float32(u[sensor_idx])
            st = torch.from_numpy(s).to(device).unsqueeze(0)
            action = actor(st).squeeze(0).cpu().numpy()
            u = np.float32(ks.advance(u, action))
            hist[t + 1] = u

    return {
        "name": name,
        "kind": "controlled",
        "checkpoint_label": ckpt_label,
        "checkpoint_path": str(actor_ckpt),
        "sensor_indices": sensor_idx.tolist(),
        "uxt": hist,
        "dt": float(ks.dt),
    }


def rollout_uncontrolled(
    init_state: np.ndarray,
    state_dim: int,
    a_dim: int,
    domain_length: float,
    max_steps: int,
) -> Dict[str, Any]:
    ks = KS(L=domain_length, N=state_dim, a_dim=a_dim)
    u = np.float32(init_state.copy())
    hist = np.zeros((max_steps + 1, state_dim), dtype=np.float32)
    hist[0] = u
    zero_action = np.zeros((a_dim,), dtype=np.float32)
    for t in range(max_steps):
        u = np.float32(ks.advance(u, zero_action))
        hist[t + 1] = u
    return {
        "name": "uncontrolled",
        "kind": "uncontrolled",
        "checkpoint_label": "none",
        "checkpoint_path": None,
        "sensor_indices": None,
        "uxt": hist,
        "dt": float(ks.dt),
    }


def plot_single_uxt(
    out_png: Path,
    out_svg: Path,
    out_pdf: Path,
    name_title: str,
    uxt: np.ndarray,
    time_axis: np.ndarray,
    x: np.ndarray,
    cmap: str,
    interpolation: str,
    vmin: float,
    vmax: float,
    dpi: int,
    title_font_size: float,
    axis_label_font_size: float,
    tick_font_size: float,
    colorbar_label_font_size: float,
    colorbar_tick_font_size: float,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    im = ax.imshow(
        uxt.T,
        origin="lower",
        aspect="auto",
        extent=[float(time_axis[0]), float(time_axis[-1]), float(x[0]), float(x[-1])],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    ax.set_xlabel("time", fontsize=float(axis_label_font_size))
    ax.set_ylabel("x grid", fontsize=float(axis_label_font_size))
    ax.set_title(name_title, loc="left", fontsize=float(title_font_size))
    ax.tick_params(axis="both", labelsize=float(tick_font_size))
    cb = fig.colorbar(im, ax=ax, pad=0.012)
    cb.set_label("u(x,t) (speed)", fontsize=float(colorbar_label_font_size))
    cb.ax.tick_params(labelsize=float(colorbar_tick_font_size))
    fig.tight_layout()
    fig.savefig(out_png, dpi=max(600, int(dpi)))
    fig.savefig(out_svg)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.state_dim <= 0:
        raise ValueError("state-dim must be positive.")
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive.")
    if not (0.0 <= args.q_low < args.q_high <= 100.0):
        raise ValueError("Quantiles must satisfy 0 <= q_low < q_high <= 100.")
    if args.title_font_size <= 0:
        raise ValueError("title-font-size must be positive.")
    if args.axis_label_font_size <= 0:
        raise ValueError("axis-label-font-size must be positive.")
    if args.tick_font_size <= 0:
        raise ValueError("tick-font-size must be positive.")
    if args.colorbar_label_font_size <= 0:
        raise ValueError("colorbar-label-font-size must be positive.")
    if args.colorbar_tick_font_size <= 0:
        raise ValueError("colorbar-tick-font-size must be positive.")

    device = resolve_device(args.device)
    x = np.loadtxt(args.x_file)
    if x.ndim != 1 or x.size != args.state_dim:
        raise ValueError(f"x-file must be shape ({args.state_dim},), got {x.shape}")

    init_states = np.loadtxt(args.init_file)
    if init_states.ndim == 1:
        init_states = init_states.reshape(1, -1)
    if init_states.ndim != 2 or init_states.shape[1] != args.state_dim:
        raise ValueError(
            f"init-file must have width {args.state_dim}, got {init_states.shape}"
        )

    rng = np.random.default_rng(args.seed)
    if args.init_row is None:
        init_row = int(rng.integers(0, init_states.shape[0]))
    else:
        init_row = int(args.init_row)
    if init_row < 0 or init_row >= init_states.shape[0]:
        raise ValueError(f"init-row out of bounds: {init_row}")
    init_state = np.float32(init_states[init_row].copy())

    specs = load_models_spec(args.models_spec)
    by_name = {str(s["name"]): s for s in specs}
    if args.model_k4_name not in by_name:
        raise ValueError(f"Model '{args.model_k4_name}' not found in models spec.")
    if args.model_k8_name not in by_name:
        raise ValueError(f"Model '{args.model_k8_name}' not found in models spec.")
    spec_k4 = by_name[args.model_k4_name]
    spec_k8 = by_name[args.model_k8_name]

    out_uncontrolled = rollout_uncontrolled(
        init_state=init_state,
        state_dim=args.state_dim,
        a_dim=int(spec_k4["a_dim"]),
        domain_length=args.domain_length,
        max_steps=args.max_steps,
    )
    out_k4 = rollout_controlled(
        spec=spec_k4,
        init_state=init_state,
        state_dim=args.state_dim,
        domain_length=args.domain_length,
        max_steps=args.max_steps,
        device=device,
    )
    out_k8 = rollout_controlled(
        spec=spec_k8,
        init_state=init_state,
        state_dim=args.state_dim,
        domain_length=args.domain_length,
        max_steps=args.max_steps,
        device=device,
    )
    outputs = [out_uncontrolled, out_k4, out_k8]

    dt = float(out_uncontrolled["dt"])
    for out in [out_k4, out_k8]:
        if abs(float(out["dt"]) - dt) > 1e-12:
            raise ValueError(f"dt mismatch: {out['name']} has {out['dt']} vs {dt}")
    time_axis = np.arange(args.max_steps + 1, dtype=np.float32) * dt

    all_vals = np.concatenate([o["uxt"].ravel() for o in outputs])
    vmin = float(np.percentile(all_vals, args.q_low))
    vmax = float(np.percentile(all_vals, args.q_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    names_and_titles = [
        ("uncontrolled_uxt", "uncontrolled"),
        ("uniform_k4_uxt", _pretty_rollout_title(out_k4["name"], out_k4["checkpoint_label"])),
        ("uniform_k8_uxt", _pretty_rollout_title(out_k8["name"], out_k8["checkpoint_label"])),
    ]

    for out, (stem, title) in zip(outputs, names_and_titles):
        plot_single_uxt(
            out_png=outdir / f"{stem}.png",
            out_svg=outdir / f"{stem}.svg",
            out_pdf=outdir / f"{stem}.pdf",
            name_title=title,
            uxt=out["uxt"],
            time_axis=time_axis,
            x=x,
            cmap=args.cmap,
            interpolation=args.interpolation,
            vmin=vmin,
            vmax=vmax,
            dpi=args.dpi,
            title_font_size=args.title_font_size,
            axis_label_font_size=args.axis_label_font_size,
            tick_font_size=args.tick_font_size,
            colorbar_label_font_size=args.colorbar_label_font_size,
            colorbar_tick_font_size=args.colorbar_tick_font_size,
        )

    summary = {
        "seed": int(args.seed),
        "init_row": int(init_row),
        "max_steps": int(args.max_steps),
        "dt": float(dt),
        "total_time": float(time_axis[-1]),
        "state_dim": int(args.state_dim),
        "domain_length": float(args.domain_length),
        "cmap": args.cmap,
        "interpolation": args.interpolation,
        "dpi_png": int(max(600, int(args.dpi))),
        "title_font_size": float(args.title_font_size),
        "axis_label_font_size": float(args.axis_label_font_size),
        "tick_font_size": float(args.tick_font_size),
        "colorbar_label_font_size": float(args.colorbar_label_font_size),
        "colorbar_tick_font_size": float(args.colorbar_tick_font_size),
        "color_scale_shared": True,
        "vmin_q": float(args.q_low),
        "vmax_q": float(args.q_high),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "device": str(device),
        "outputs": {
            "uncontrolled": {
                "png": str(outdir / "uncontrolled_uxt.png"),
                "svg": str(outdir / "uncontrolled_uxt.svg"),
                "pdf": str(outdir / "uncontrolled_uxt.pdf"),
            },
            "uniform_k4": {
                "png": str(outdir / "uniform_k4_uxt.png"),
                "svg": str(outdir / "uniform_k4_uxt.svg"),
                "pdf": str(outdir / "uniform_k4_uxt.pdf"),
                "checkpoint": out_k4["checkpoint_label"],
            },
            "uniform_k8": {
                "png": str(outdir / "uniform_k8_uxt.png"),
                "svg": str(outdir / "uniform_k8_uxt.svg"),
                "pdf": str(outdir / "uniform_k8_uxt.pdf"),
                "checkpoint": out_k8["checkpoint_label"],
            },
        },
    }
    (outdir / "rollout_triplet_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] Generated rollout triplet in: {outdir}")
    print(f"INIT row used: {init_row}")


if __name__ == "__main__":
    main()
