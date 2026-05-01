from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
PLACEMENT_STUDY_DIR = ROOT / "studies" / "sensor_placement"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ks_control.ks import KS
from ks_control.placement_helpers import (
    build_forbidden_indices,
    build_uniform_indices,
    corridor_ranges,
    validate_indices,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate baseline sensor placement layouts for the active sensor-placement study."
    )
    ap.add_argument("--sensor-counts", type=str, default="4,8,12,16,20")
    ap.add_argument("--strategies", type=str, default="uniform")
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--a-dim", type=int, default=4)
    ap.add_argument("--x-file", type=str, default=str(DATA_DIR / "x.dat"))
    ap.add_argument("--domain-length", type=float, default=22.0)
    ap.add_argument("--exclude-window", type=int, default=2)
    periodic_group = ap.add_mutually_exclusive_group()
    periodic_group.add_argument("--periodic", dest="periodic", action="store_true")
    periodic_group.add_argument("--no-periodic", dest="periodic", action="store_false")
    ap.set_defaults(periodic=True)
    ap.add_argument("--plot-script", type=str, default=str(ROOT / "scripts" / "plot_placement.py"))
    ap.add_argument("--study-root", type=str, default=str(PLACEMENT_STUDY_DIR))
    ap.add_argument("--manifest-name", type=str, default="generated_layouts.json")
    ap.add_argument("--no-show", action="store_true")
    return ap.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def parse_csv_int(expr: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for raw in expr.split(","):
        tok = raw.strip()
        if not tok:
            continue
        val = int(tok)
        if val not in seen:
            seen.add(val)
            out.append(val)
    if not out:
        raise ValueError("No integer values parsed.")
    return out


def parse_csv_str(expr: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in expr.split(","):
        tok = raw.strip()
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    if not out:
        raise ValueError("No strategy values parsed.")
    return out


def run_plot(
    plot_script: Path,
    sensor_json: Path,
    x_file: Path,
    a_dim: int,
    domain_length: float,
    output_plot: Path,
    output_json: Path,
    no_show: bool,
) -> None:
    cmd = [
        sys.executable,
        str(plot_script),
        "--sensor-indices",
        str(sensor_json),
        "--x-file",
        str(x_file),
        "--a-dim",
        str(a_dim),
        "--domain-length",
        str(domain_length),
        "--output-plot",
        str(output_plot),
        "--output-json",
        str(output_json),
    ]
    if no_show:
        cmd.append("--no-show")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    sensor_counts = parse_csv_int(args.sensor_counts)
    strategies = parse_csv_str(args.strategies)
    unsupported = [strategy for strategy in strategies if strategy != "uniform"]
    if unsupported:
        raise ValueError(
            f"Unsupported placement strategies in the live workflow: {unsupported}. "
            "Add new strategies explicitly as the sensor-placement study progresses."
        )

    x = np.loadtxt(args.x_file)
    if x.ndim != 1:
        raise ValueError(f"x-file must be 1D, got shape {x.shape}")
    if x.size != args.state_dim:
        raise ValueError(f"x-file length {x.size} does not match state-dim {args.state_dim}")

    ks = KS(L=args.domain_length, N=args.state_dim, a_dim=args.a_dim)
    actuator_indices = [int(np.argmax(ks.B[:, i])) for i in range(args.a_dim)]
    forbidden = set(
        build_forbidden_indices(
            actuator_indices=actuator_indices,
            state_dim=args.state_dim,
            window=args.exclude_window,
            periodic=args.periodic,
        )
    )
    corridors = corridor_ranges(actuator_indices, state_dim=args.state_dim, window=args.exclude_window)

    study_root = Path(args.study_root)
    layouts_dir = study_root / "layouts"
    manifests_dir = study_root / "manifests"
    plots_dir = study_root / "results" / "placement_plots"
    layouts_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_script = Path(args.plot_script)
    manifest_layouts: List[Dict[str, Any]] = []

    for strategy in strategies:
        if strategy != "uniform":
            continue
        for k in sensor_counts:
            indices = build_uniform_indices(k=k, corridor_safe_ranges=corridors, state_dim=args.state_dim)
            validate_indices(indices=indices, k=k, forbidden=forbidden, state_dim=args.state_dim)

            layout_name = f"{strategy}_k{k}"
            layout_path = layouts_dir / f"{layout_name}.json"
            plot_path = plots_dir / f"{layout_name}.png"
            plot_json = plots_dir / f"{layout_name}.json"

            layout_payload = {
                "name": layout_name,
                "strategy": strategy,
                "k": k,
                "sensor_indices": indices,
            }
            layout_path.write_text(json.dumps(layout_payload, indent=2), encoding="utf-8")
            run_plot(
                plot_script=plot_script,
                sensor_json=layout_path,
                x_file=Path(args.x_file),
                a_dim=args.a_dim,
                domain_length=args.domain_length,
                output_plot=plot_path,
                output_json=plot_json,
                no_show=args.no_show,
            )

            manifest_layouts.append(
                {
                    "name": layout_name,
                    "strategy": strategy,
                    "k": k,
                    "sensor_indices": indices,
                    "layout_json": to_repo_relative(layout_path),
                    "plot_png": to_repo_relative(plot_path),
                    "plot_json": to_repo_relative(plot_json),
                }
            )

    manifest = {
        "study": "sensor_placement",
        "generated_at": now_iso(),
        "state_dim": args.state_dim,
        "a_dim": args.a_dim,
        "sensor_counts": sensor_counts,
        "placement_strategies": strategies,
        "domain_length": args.domain_length,
        "actuator_indices": actuator_indices,
        "exclude_window": args.exclude_window,
        "periodic": args.periodic,
        "layouts": manifest_layouts,
    }
    manifest_path = manifests_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote placement manifest to {manifest_path}")


if __name__ == "__main__":
    main()
