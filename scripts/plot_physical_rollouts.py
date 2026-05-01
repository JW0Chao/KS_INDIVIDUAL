from __future__ import annotations

import argparse
import json
import os
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

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ks_control.ks import KS
from ks_control import model
from ks_control.evaluation_helpers import resolve_sensor_indices


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Plot physical rollouts u(x,t) for multiple trained models using the same "
            "initial condition sampled from INIT.dat."
        )
    )
    ap.add_argument("--models-spec", type=str, required=True, help="Path to models spec JSON.")
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--domain-length", type=float, default=22.0)
    ap.add_argument("--x-file", type=str, default=str(DATA_DIR / "x.dat"))
    ap.add_argument("--init-file", type=str, default=str(DATA_DIR / "INIT.dat"))
    ap.add_argument("--seed", type=int, default=0, help="Seed for random INIT-row selection.")
    ap.add_argument(
        "--init-row",
        type=int,
        default=None,
        help="Optional explicit INIT row index. If omitted, one row is sampled with --seed.",
    )
    ap.add_argument("--max-steps", type=int, default=1500, help="Rollout steps.")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--cmap", type=str, default="viridis")
    ap.add_argument(
        "--interpolation",
        type=str,
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        help="Image interpolation mode for smoother heatmaps.",
    )
    ap.add_argument(
        "--no-shared-scale",
        action="store_true",
        help="Use per-model color scale instead of one shared scale for all models.",
    )
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--outdir", type=str, default=str(SENSOR_COUNT_RESULTS_DIR / "physical_rollouts"))
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
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("models-spec must be a non-empty list or {'models': [...]} object.")

    out: List[Dict[str, Any]] = []
    for i, m in enumerate(data):
        if not isinstance(m, dict):
            raise ValueError(f"models-spec entry #{i} must be an object.")
        for req in ["name", "models_dir", "s_dim", "a_dim", "a_max"]:
            if req not in m:
                raise ValueError(f"models-spec entry #{i} missing required field '{req}'.")
        out.append(m)
    return out


def pick_actor_checkpoint(models_dir: Path, requested_episode: Optional[int]) -> Tuple[Path, str]:
    if requested_episode is not None:
        p = models_dir / f"{requested_episode}_actor.pt"
        if p.exists():
            return p, str(int(requested_episode))
        best = models_dir / "best_actor.pt"
        if best.exists():
            return best, f"best(fallback_for_{int(requested_episode)})"
        raise FileNotFoundError(
            f"Missing requested actor checkpoint '{p}' and no '{best}' fallback."
        )

    actor_ckpts = [
        f for f in os.listdir(models_dir)
        if f.endswith("_actor.pt") and not f.endswith("_target_actor.pt")
    ]
    numeric = []
    for f in actor_ckpts:
        stem = f.split("_")[0]
        if stem.isdigit():
            numeric.append(int(stem))
    if numeric:
        ep = max(numeric)
        return models_dir / f"{ep}_actor.pt", str(ep)

    best = models_dir / "best_actor.pt"
    if best.exists():
        return best, "best"
    raise FileNotFoundError(f"No actor checkpoint found in {models_dir}")


def rollout_uxt(
    spec: Dict[str, Any],
    init_state: np.ndarray,
    x: np.ndarray,
    domain_length: float,
    max_steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    name = str(spec["name"])
    models_dir = Path(str(spec["models_dir"]))
    s_dim = int(spec["s_dim"])
    a_dim = int(spec["a_dim"])
    a_max = float(spec["a_max"])
    sensor_indices = resolve_sensor_indices(spec, x.size)

    ckpt_path, ckpt_label = pick_actor_checkpoint(models_dir, spec.get("episode", None))
    actor = model.Actor(s_dim, a_dim, a_max).to(device)
    actor.load_state_dict(torch.load(ckpt_path, map_location=device))
    actor.eval()

    ks = KS(L=domain_length, N=x.size, a_dim=a_dim)
    dt = float(ks.dt)
    u = np.float32(init_state.copy())
    hist = np.zeros((max_steps + 1, x.size), dtype=np.float32)
    hist[0] = u

    with torch.no_grad():
        for t in range(max_steps):
            s = np.float32(u[sensor_indices])
            st = torch.from_numpy(s).to(device).unsqueeze(0)
            at = actor(st)
            action = at.squeeze(0).cpu().numpy()
            u = np.float32(ks.advance(u, action))
            hist[t + 1] = u

    return {
        "name": name,
        "checkpoint_label": ckpt_label,
        "checkpoint_path": str(ckpt_path),
        "sensor_indices": sensor_indices,
        "dt": dt,
        "uxt": hist,
    }


def plot_single(
    outpath: Path,
    name: str,
    uxt: np.ndarray,
    time_axis: np.ndarray,
    x: np.ndarray,
    init_row: int,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    dpi: int,
    interpolation: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
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
    ax.set_xlabel("time")
    ax.set_ylabel("x grid")
    ax.set_title(f"{name}: rollout of u(x,t), INIT row {init_row}")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("u(x,t) (speed)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_panel(
    outpath: Path,
    outputs: List[Dict[str, Any]],
    time_axis: np.ndarray,
    x: np.ndarray,
    init_row: int,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    dpi: int,
    interpolation: str,
) -> None:
    n = len(outputs)
    fig, axes = plt.subplots(
        n, 1, figsize=(11, max(2.6 * n, 4.0)), sharex=True, constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    last_im = None
    for ax, out in zip(axes, outputs):
        last_im = ax.imshow(
            out["uxt"].T,
            origin="lower",
            aspect="auto",
            extent=[float(time_axis[0]), float(time_axis[-1]), float(x[0]), float(x[-1])],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )
        ax.set_ylabel("x grid")
        ax.set_title(f"{out['name']} (ckpt: {out['checkpoint_label']})", loc="left")
    axes[-1].set_xlabel("time")
    if last_im is not None:
        cb = fig.colorbar(last_im, ax=axes, shrink=0.98, pad=0.015)
        cb.set_label("u(x,t) (speed)")
    fig.suptitle(f"Physical Rollout Heatmaps u(x,t) with shared INIT row {init_row}", y=1.01)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x = np.loadtxt(args.x_file)
    if x.ndim != 1:
        raise ValueError(f"x-file must be 1D, got {x.shape}")
    if x.size != args.state_dim:
        raise ValueError(f"x size {x.size} != --state-dim {args.state_dim}")

    init_states = np.loadtxt(args.init_file)
    if init_states.ndim == 1:
        init_states = init_states.reshape(1, -1)
    if init_states.ndim != 2 or init_states.shape[1] != args.state_dim:
        raise ValueError(
            f"init-file must have shape (n_rows, {args.state_dim}), got {init_states.shape}"
        )

    rng = np.random.default_rng(args.seed)
    if args.init_row is None:
        init_row = int(rng.integers(0, init_states.shape[0]))
    else:
        init_row = int(args.init_row)
    if init_row < 0 or init_row >= init_states.shape[0]:
        raise ValueError(f"init-row out of bounds: {init_row} for {init_states.shape[0]} rows")
    init_state = np.float32(init_states[init_row].copy())

    specs = load_models_spec(args.models_spec)
    outputs: List[Dict[str, Any]] = []
    for spec in specs:
        out = rollout_uxt(
            spec=spec,
            init_state=init_state,
            x=x,
            domain_length=args.domain_length,
            max_steps=args.max_steps,
            device=device,
        )
        outputs.append(out)

    if not outputs:
        raise ValueError("No model outputs generated.")

    dt = float(outputs[0]["dt"])
    for out in outputs[1:]:
        if abs(float(out["dt"]) - dt) > 1e-12:
            raise ValueError(f"dt mismatch across models: {out['name']} has dt={out['dt']} vs {dt}")
    time_axis = np.arange(args.max_steps + 1, dtype=np.float32) * dt

    if args.vmin is not None or args.vmax is not None:
        vmin = args.vmin
        vmax = args.vmax
    elif args.no_shared_scale:
        vmin = None
        vmax = None
    else:
        vmin = min(float(np.min(o["uxt"])) for o in outputs)
        vmax = max(float(np.max(o["uxt"])) for o in outputs)

    # Per-model plots.
    for out in outputs:
        one_vmin = vmin
        one_vmax = vmax
        if args.no_shared_scale and args.vmin is None and args.vmax is None:
            one_vmin = float(np.min(out["uxt"]))
            one_vmax = float(np.max(out["uxt"]))
        plot_single(
            outpath=outdir / f"rollout_uxt_{out['name']}.png",
            name=out["name"],
            uxt=out["uxt"],
            time_axis=time_axis,
            x=x,
            init_row=init_row,
            cmap=args.cmap,
            vmin=one_vmin,
            vmax=one_vmax,
            dpi=args.dpi,
            interpolation=args.interpolation,
        )

    # Combined panel plot.
    plot_panel(
        outpath=outdir / "rollout_uxt_all_models.png",
        outputs=outputs,
        time_axis=time_axis,
        x=x,
        init_row=init_row,
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
        dpi=args.dpi,
        interpolation=args.interpolation,
    )

    # Save raw arrays for reproducibility / reuse.
    arrays: Dict[str, Any] = {
        "x": x.astype(np.float32),
        "time_axis": time_axis.astype(np.float32),
        "init_row": np.int64(init_row),
        "init_state": init_state.astype(np.float32),
    }
    for i, out in enumerate(outputs):
        key = f"{i}_{out['name']}"
        arrays[f"{key}__uxt"] = out["uxt"]
        arrays[f"{key}__sensor_indices"] = out["sensor_indices"].astype(np.int64)
        arrays[f"{key}__dt"] = np.float32(out["dt"])
    np.savez_compressed(outdir / "rollout_uxt_arrays.npz", **arrays)

    summary = {
        "models_spec": args.models_spec,
        "init_file": args.init_file,
        "init_row": int(init_row),
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "domain_length": float(args.domain_length),
        "device": str(device),
        "dt": dt,
        "shared_color_scale": bool(not args.no_shared_scale),
        "interpolation": args.interpolation,
        "vmin": None if vmin is None else float(vmin),
        "vmax": None if vmax is None else float(vmax),
        "plots": [f"rollout_uxt_{o['name']}.png" for o in outputs] + ["rollout_uxt_all_models.png"],
        "models": [
            {
                "name": o["name"],
                "checkpoint_label": o["checkpoint_label"],
                "checkpoint_path": o["checkpoint_path"],
                "sensor_indices": o["sensor_indices"].tolist(),
            }
            for o in outputs
        ],
    }
    (outdir / "rollout_uxt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] Generated rollout u(x,t) plots for {len(outputs)} models.")
    print(f"INIT row used: {init_row}")
    print(f"Saved panel: {outdir / 'rollout_uxt_all_models.png'}")
    print(f"Saved summary: {outdir / 'rollout_uxt_summary.json'}")


if __name__ == "__main__":
    main()
