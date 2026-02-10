from __future__ import annotations

import argparse
import math
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from KS import KS


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_outputs(outputs: Optional[Sequence[Path]]) -> str:
    if not outputs:
        return "- Outputs: none\n"
    lines = ["- Outputs:\n"]
    for out in outputs:
        lines.append(f"  - `{out}`\n")
    return "".join(lines)


class StepLogger:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.step = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            f"# Steps Taken\n\n"
            f"- Run ID: `{run_id}`\n"
            f"- Started (UTC): `{_iso_now()}`\n\n",
            encoding="utf-8",
        )

    def _append(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text)

    def log_note(self, stage: str, message: str, outputs: Optional[Sequence[Path]] = None) -> None:
        self.step += 1
        entry = (
            f"## Step {self.step}: {stage}\n"
            f"- Type: note\n"
            f"- Timestamp (UTC): `{_iso_now()}`\n"
            f"- Message: {message}\n"
            f"{_format_outputs(outputs)}\n"
        )
        self._append(entry)

    def run_command(
        self,
        stage: str,
        command: Sequence[str] | str,
        workdir: Path,
        outputs: Optional[Sequence[Path]] = None,
    ) -> None:
        start_iso = _iso_now()
        t0 = time.perf_counter()
        if isinstance(command, str):
            cmd_text = command
            proc = subprocess.run(command, cwd=workdir, shell=True)
        else:
            cmd_text = subprocess.list2cmdline(list(command))
            proc = subprocess.run(list(command), cwd=workdir, shell=False)
        t1 = time.perf_counter()
        end_iso = _iso_now()
        duration = t1 - t0

        self.step += 1
        entry = (
            f"## Step {self.step}: {stage}\n"
            f"- Type: command\n"
            f"- Start (UTC): `{start_iso}`\n"
            f"- End (UTC): `{end_iso}`\n"
            f"- Duration (s): `{duration:.3f}`\n"
            f"- Exit code: `{proc.returncode}`\n"
            f"- Command: `{cmd_text}`\n"
            f"{_format_outputs(outputs)}\n"
        )
        self._append(entry)

        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {proc.returncode}: {cmd_text}")

    def finalize(self, status: str, summary_outputs: Sequence[Path]) -> None:
        section = (
            f"## Run Summary\n"
            f"- Finished (UTC): `{_iso_now()}`\n"
            f"- Status: `{status}`\n"
            f"{_format_outputs(summary_outputs)}\n"
        )
        self._append(section)


def _parse_index_expr(expr: str) -> List[int]:
    out: List[int] = []
    seen: Set[int] = set()
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


def parse_sensor_indices(expr_or_path: str, state_dim: int) -> List[int]:
    p = Path(expr_or_path)
    if p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "indices" in data:
                data = data["indices"]
            else:
                raise ValueError("Sensor index JSON dict must contain key 'indices'.")
        if not isinstance(data, list):
            raise ValueError("Sensor index JSON must be a list.")
        vals = []
        seen: Set[int] = set()
        for item in data:
            if isinstance(item, str) and item.startswith("s"):
                idx = int(item[1:])
            else:
                idx = int(item)
            if idx not in seen:
                seen.add(idx)
                vals.append(idx)
    else:
        vals = _parse_index_expr(expr_or_path)

    bad = [i for i in vals if i < 0 or i >= state_dim]
    if bad:
        raise ValueError(f"Sensor indices out of bounds for state_dim={state_dim}: {bad}")
    if len(set(vals)) != len(vals):
        raise ValueError("Sensor indices contain duplicates.")
    return vals


def make_random_sensor_sets(
    state_dim: int,
    set_size: int,
    count: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    if set_size <= 0:
        raise ValueError("random-k must be positive")
    if set_size > state_dim:
        raise ValueError(f"random-k={set_size} cannot exceed state_dim={state_dim}")
    target = min(count, int(math.comb(state_dim, set_size)))
    out: List[List[int]] = []
    seen: Set[tuple[int, ...]] = set()
    while len(out) < target:
        idx = tuple(sorted(rng.choice(state_dim, size=set_size, replace=False).tolist()))
        if idx in seen:
            continue
        seen.add(idx)
        out.append(list(idx))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run research pipeline stages with time-ordered logging to results/<run_id>/Steps_taken.md."
    )
    ap.add_argument("--results-root", type=str, default="results")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--workdir", type=str, default=".")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--x-file", type=str, default="x.dat")
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--domain-length", type=float, default=22.0)
    ap.add_argument("--a-dim", type=int, default=4)
    ap.add_argument("--dt-expected", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--teacher-sensor-indices", type=str, default="0-63")
    ap.add_argument("--bucci-sensor-indices", type=str, default="4,12,20,28,36,44,52,60")
    ap.add_argument("--shap-sensor-indices", type=str, default=None)
    ap.add_argument(
        "--shap-topk-json-template",
        type=str,
        default=None,
        help=(
            "Optional template path to SHAP top-k JSON (e.g. "
            "'{run_dir}/shap/fine_g1_from_g8/topk_actuator_excluded.json'). "
            "Used to auto-load SHAP sensor indices after Stage 5 when --shap-sensor-indices is omitted."
        ),
    )
    ap.add_argument("--random-k", type=int, default=8)
    ap.add_argument("--random-count", type=int, default=3)

    ap.add_argument("--smoke", action="store_true", help="Short run: audit + minimal plotting stages only.")
    ap.add_argument("--inject-failure-cmd", type=str, default=None, help="Optional command to intentionally fail.")

    ap.add_argument("--teacher-train-cmd", type=str, default=None)
    ap.add_argument("--shap-discovery-cmd", type=str, default=None)
    ap.add_argument("--bucci-train-cmd", type=str, default=None)
    ap.add_argument("--shap-train-cmd", type=str, default=None)
    ap.add_argument(
        "--random-train-cmd-template",
        type=str,
        default=None,
        help="Template with {random_idx} and {sensor_indices_csv}.",
    )
    ap.add_argument("--evaluation-cmd", type=str, default=None)
    return ap.parse_args()


def _render_template(template: str, context: Dict[str, Any]) -> str:
    try:
        return template.format(**context)
    except KeyError as exc:
        raise ValueError(f"Missing template key {exc} in command template: {template}") from exc


def _build_models_spec(
    run_dir: Path,
    a_dim: int,
    default_a_max: float,
    bucci_indices: List[int],
    shap_indices: Optional[List[int]],
    random_sets: List[List[int]],
) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []

    teacher_dir = run_dir / "models" / "teacher64"
    if teacher_dir.exists():
        models.append(
            {
                "name": "Full-64",
                "role": "full64",
                "models_dir": str(teacher_dir),
                "s_dim": 64,
                "a_dim": a_dim,
                "a_max": default_a_max,
            }
        )

    bucci_dir = run_dir / "models" / "bucci8"
    if bucci_dir.exists():
        models.append(
            {
                "name": "Bucci-8",
                "role": "bucci8",
                "models_dir": str(bucci_dir),
                "s_dim": len(bucci_indices),
                "a_dim": a_dim,
                "a_max": default_a_max,
                "sensor_indices": bucci_indices,
            }
        )

    shap_dir = run_dir / "models" / "shap8"
    if shap_dir.exists() and shap_indices is not None:
        models.append(
            {
                "name": "SHAP-8",
                "role": "shap8",
                "models_dir": str(shap_dir),
                "s_dim": len(shap_indices),
                "a_dim": a_dim,
                "a_max": default_a_max,
                "sensor_indices": shap_indices,
            }
        )

    for ridx, rset in enumerate(random_sets):
        rdir = run_dir / "models" / f"random8_r{ridx}"
        if not rdir.exists():
            continue
        models.append(
            {
                "name": f"Random-8-r{ridx}",
                "role": "other",
                "models_dir": str(rdir),
                "s_dim": len(rset),
                "a_dim": a_dim,
                "a_max": default_a_max,
                "sensor_indices": rset,
            }
        )

    return models


def _plot_layout(
    logger: StepLogger,
    args: argparse.Namespace,
    workdir: Path,
    run_dir: Path,
    stage_name: str,
    sensor_indices: List[int],
    stem: str,
) -> List[Path]:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_plot = plots_dir / f"{stem}.png"
    out_json = plots_dir / f"{stem}.json"
    cmd = [
        args.python,
        "plot_sensor_actuator_positions.py",
        "--sensor-indices",
        ",".join(str(i) for i in sensor_indices),
        "--x-file",
        args.x_file,
        "--a-dim",
        str(args.a_dim),
        "--domain-length",
        str(args.domain_length),
        "--overlay-forcing",
        "--output-plot",
        str(out_plot),
        "--output-json",
        str(out_json),
        "--no-show",
    ]
    logger.run_command(stage=stage_name, command=cmd, workdir=workdir, outputs=[out_plot, out_json])
    return [out_plot, out_json]


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (workdir / args.results_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = StepLogger(path=run_dir / "Steps_taken.md", run_id=run_id)
    summary_outputs: List[Path] = [run_dir / "Steps_taken.md"]
    status = "success"

    try:
        rng = np.random.default_rng(args.seed)

        # Stage 1: Geometry/invariants audit.
        x = np.loadtxt(workdir / args.x_file)
        if x.ndim != 1:
            raise ValueError(f"x-file must be 1D, got shape {x.shape}")
        if x.shape[0] != args.state_dim:
            raise ValueError(
                f"state-dim mismatch with x-file: state_dim={args.state_dim}, x-size={x.shape[0]}"
            )
        ks = KS(L=args.domain_length, N=args.state_dim, a_dim=args.a_dim)
        actuator_indices = [int(np.argmax(ks.B[:, i])) for i in range(args.a_dim)]
        expected_bucci_actuators = [int(i * args.state_dim / args.a_dim) for i in range(args.a_dim)]
        bucci_indices = parse_sensor_indices(args.bucci_sensor_indices, state_dim=args.state_dim)
        invariants = {
            "N": int(args.state_dim),
            "L": float(args.domain_length),
            "dx": float(args.domain_length / args.state_dim),
            "dt": float(ks.dt),
            "dt_expected": float(args.dt_expected),
            "dt_match": bool(abs(float(ks.dt) - args.dt_expected) <= 1e-12),
            "a_dim": int(args.a_dim),
            "actuator_indices_current_code": actuator_indices,
            "actuator_indices_expected_bucci": expected_bucci_actuators,
            "actuator_match_expected_bucci": bool(actuator_indices == expected_bucci_actuators),
            "bucci_8_sensor_indices": bucci_indices,
            "seed": int(args.seed),
            "x_file": args.x_file,
        }
        invariants_path = run_dir / "invariants_audit.json"
        invariants_path.write_text(json.dumps(invariants, indent=2), encoding="utf-8")
        summary_outputs.append(invariants_path)
        logger.log_note(
            stage="Stage 1 - Geometry & Invariants Audit",
            message="Computed and saved invariant audit JSON.",
            outputs=[invariants_path],
        )
        if actuator_indices != expected_bucci_actuators:
            raise ValueError(
                "Actuator geometry check failed. Expected fixed indices "
                f"{expected_bucci_actuators}, got {actuator_indices}."
            )

        if args.inject_failure_cmd:
            logger.run_command(
                stage="Injected Failure Command",
                command=args.inject_failure_cmd,
                workdir=workdir,
                outputs=None,
            )

        # Stage 2: Layout definitions.
        teacher_indices = parse_sensor_indices(args.teacher_sensor_indices, state_dim=args.state_dim)
        shap_indices = (
            parse_sensor_indices(args.shap_sensor_indices, state_dim=args.state_dim)
            if args.shap_sensor_indices
            else None
        )
        random_sets = make_random_sensor_sets(
            state_dim=args.state_dim,
            set_size=args.random_k,
            count=args.random_count,
            rng=rng,
        )
        layouts = {
            "teacher64": teacher_indices,
            "bucci8": bucci_indices,
            "shap8": shap_indices,
            "random8": random_sets,
        }
        layouts_path = run_dir / "sensor_layouts.json"
        layouts_path.write_text(json.dumps(layouts, indent=2), encoding="utf-8")
        summary_outputs.append(layouts_path)
        logger.log_note(
            stage="Stage 2 - Sensor Layout Assembly",
            message="Prepared teacher/Bucci/SHAP/random sensor layouts.",
            outputs=[layouts_path],
        )

        # Stage 3: Position + forcing plots.
        plot_outputs: List[Path] = []
        plot_outputs.extend(
            _plot_layout(
                logger=logger,
                args=args,
                workdir=workdir,
                run_dir=run_dir,
                stage_name="Stage 3 - Plot Teacher64 Layout",
                sensor_indices=teacher_indices,
                stem="positions_teacher64",
            )
        )
        if args.smoke:
            logger.log_note(
                stage="Smoke Mode",
                message="Smoke mode enabled: skipping remaining full-pipeline stages.",
                outputs=None,
            )
        else:
            plot_outputs.extend(
                _plot_layout(
                    logger=logger,
                    args=args,
                    workdir=workdir,
                    run_dir=run_dir,
                    stage_name="Stage 3 - Plot Bucci8 Layout",
                    sensor_indices=bucci_indices,
                    stem="positions_bucci8",
                )
            )
            if shap_indices is not None:
                plot_outputs.extend(
                    _plot_layout(
                        logger=logger,
                        args=args,
                        workdir=workdir,
                        run_dir=run_dir,
                        stage_name="Stage 3 - Plot SHAP8 Layout",
                        sensor_indices=shap_indices,
                        stem="positions_shap8",
                    )
                )
            else:
                logger.log_note(
                    stage="Stage 3 - Plot SHAP8 Layout",
                    message="Skipped: no --shap-sensor-indices provided.",
                    outputs=None,
                )
            for ridx, rset in enumerate(random_sets):
                plot_outputs.extend(
                    _plot_layout(
                        logger=logger,
                        args=args,
                        workdir=workdir,
                        run_dir=run_dir,
                        stage_name=f"Stage 3 - Plot Random8 Layout r{ridx}",
                        sensor_indices=rset,
                        stem=f"positions_random8_r{ridx}",
                    )
                )

        summary_outputs.extend(plot_outputs)

        # Stage commands (full mode only).
        if not args.smoke:
            context: Dict[str, Any] = {
                "run_dir": str(run_dir),
                "python": args.python,
                "teacher_sensor_indices_csv": ",".join(str(i) for i in teacher_indices),
                "bucci_sensor_indices_csv": ",".join(str(i) for i in bucci_indices),
                "shap_sensor_indices_csv": ",".join(str(i) for i in shap_indices) if shap_indices is not None else "",
                "layouts_json": str(layouts_path),
            }

            if args.teacher_train_cmd:
                cmd = _render_template(args.teacher_train_cmd, context)
                logger.run_command("Stage 4 - Teacher64 Training", cmd, workdir, outputs=None)
            else:
                logger.log_note("Stage 4 - Teacher64 Training", "Skipped: no --teacher-train-cmd provided.")

            if args.shap_discovery_cmd:
                cmd = _render_template(args.shap_discovery_cmd, context)
                logger.run_command("Stage 5 - SHAP Region Discovery", cmd, workdir, outputs=None)

                if shap_indices is None and args.shap_topk_json_template:
                    shap_topk_path = Path(_render_template(args.shap_topk_json_template, context))
                    if not shap_topk_path.exists():
                        raise FileNotFoundError(
                            "shap-topk-json-template was provided but file does not exist after Stage 5: "
                            f"{shap_topk_path}"
                        )
                    shap_indices = parse_sensor_indices(str(shap_topk_path), state_dim=args.state_dim)
                    context["shap_sensor_indices_csv"] = ",".join(str(i) for i in shap_indices)
                    logger.log_note(
                        stage="Stage 5b - Load SHAP Sensor Indices",
                        message="Loaded SHAP top-k sensor indices from Stage 5 output.",
                        outputs=[shap_topk_path],
                    )

                    plot_outputs.extend(
                        _plot_layout(
                            logger=logger,
                            args=args,
                            workdir=workdir,
                            run_dir=run_dir,
                            stage_name="Stage 5c - Plot SHAP8 Layout (from SHAP discovery)",
                            sensor_indices=shap_indices,
                            stem="positions_shap8",
                        )
                    )
            else:
                logger.log_note("Stage 5 - SHAP Region Discovery", "Skipped: no --shap-discovery-cmd provided.")

            if args.bucci_train_cmd:
                cmd = _render_template(args.bucci_train_cmd, context)
                logger.run_command("Stage 6 - Bucci8 Training", cmd, workdir, outputs=None)
            else:
                logger.log_note("Stage 6 - Bucci8 Training", "Skipped: no --bucci-train-cmd provided.")

            if args.shap_train_cmd:
                if shap_indices is None:
                    raise ValueError(
                        "shap-train-cmd was provided but SHAP sensor indices are unavailable. "
                        "Provide --shap-sensor-indices or --shap-topk-json-template."
                    )
                cmd = _render_template(args.shap_train_cmd, context)
                logger.run_command("Stage 7 - SHAP8 Training", cmd, workdir, outputs=None)
            else:
                logger.log_note("Stage 7 - SHAP8 Training", "Skipped: no --shap-train-cmd provided.")

            if args.random_train_cmd_template:
                for ridx, rset in enumerate(random_sets):
                    random_context = dict(context)
                    random_context.update(
                        {
                            "random_idx": ridx,
                            "sensor_indices_csv": ",".join(str(i) for i in rset),
                        }
                    )
                    cmd = _render_template(args.random_train_cmd_template, random_context)
                    logger.run_command(f"Stage 8 - Random8 Training r{ridx}", cmd, workdir, outputs=None)
            else:
                logger.log_note("Stage 8 - Random8 Training", "Skipped: no --random-train-cmd-template provided.")

            models_spec = _build_models_spec(
                run_dir=run_dir,
                a_dim=args.a_dim,
                default_a_max=0.5,
                bucci_indices=bucci_indices,
                shap_indices=shap_indices,
                random_sets=random_sets,
            )
            models_spec_path = run_dir / "models_spec_pipeline.json"
            models_spec_path.write_text(json.dumps({"models": models_spec}, indent=2), encoding="utf-8")
            context["models_spec_path"] = str(models_spec_path)
            summary_outputs.append(models_spec_path)
            logger.log_note(
                stage="Stage 8b - Build Models Spec",
                message="Wrote models spec for pipeline evaluation stage.",
                outputs=[models_spec_path],
            )

            if args.evaluation_cmd:
                cmd = _render_template(args.evaluation_cmd, context)
                logger.run_command("Stage 9 - Evaluation", cmd, workdir, outputs=None)
            else:
                logger.log_note("Stage 9 - Evaluation", "Skipped: no --evaluation-cmd provided.")

    except Exception as exc:
        status = "failed"
        logger.log_note(
            stage="Pipeline Failure",
            message=f"{type(exc).__name__}: {exc}",
            outputs=None,
        )
        logger.finalize(status=status, summary_outputs=summary_outputs)
        raise

    logger.finalize(status=status, summary_outputs=summary_outputs)
    print(f"Run complete. Artifacts: {run_dir}")
    print(f"Step log: {run_dir / 'Steps_taken.md'}")


if __name__ == "__main__":
    main()
