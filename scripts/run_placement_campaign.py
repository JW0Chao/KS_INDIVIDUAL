from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PLACEMENT_STUDY_DIR = ROOT / "studies" / "sensor_placement"
STUDY_PROTOCOL_DIR = ROOT / "studies" / "controller_protocol"
DEFAULT_BUNDLE_ROOT = ROOT / "artifacts" / "raw" / "sensor_placement" / "training_bundle"
DEFAULT_INIT_SPLIT_FILE = STUDY_PROTOCOL_DIR / "manifests" / "controller_setup_split.json"

NUMERIC_ACTOR_SUFFIX = "_actor.pt"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train a campaign of placement-study layouts from a generic placement manifest."
    )
    ap.add_argument(
        "--placements-manifest",
        type=str,
        default=str(PLACEMENT_STUDY_DIR / "manifests" / "generated_layouts.json"),
    )
    ap.add_argument("--bundle-root", type=str, default=str(DEFAULT_BUNDLE_ROOT))
    ap.add_argument("--python-bin", type=str, default=sys.executable)
    ap.add_argument("--train-script", type=str, default=str(ROOT / "scripts" / "train_run.py"))

    ap.add_argument("--setup-name", type=str, default=None, help="Optional setup-name override. Default: layout name.")
    ap.add_argument("--target-file", type=str, default=str(DATA_DIR / "u3.dat"))
    ap.add_argument("--x-file", type=str, default=str(DATA_DIR / "x.dat"))
    ap.add_argument("--init-file", type=str, default=str(DATA_DIR / "INIT.dat"))
    ap.add_argument("--init-split-file", type=str, default=str(DEFAULT_INIT_SPLIT_FILE))
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument("--train-split-size", type=int, default=20)
    ap.add_argument("--val-split-size", type=int, default=20)
    ap.add_argument("--test-split-size", type=int, default=30)
    ap.add_argument("--action-dim", type=int, default=4)
    ap.add_argument("--action-lim", type=float, default=1.0)
    ap.add_argument("--domain-length", type=float, default=22.0)
    ap.add_argument("--max-total-reward", type=float, default=-35.0)

    ap.add_argument("--max-episodes", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--save-every", type=int, default=10)

    early_group = ap.add_mutually_exclusive_group()
    early_group.add_argument("--early-stop-enabled", dest="early_stop_enabled", action="store_true")
    early_group.add_argument("--no-early-stop", dest="early_stop_enabled", action="store_false")
    ap.set_defaults(early_stop_enabled=True)

    ap.add_argument("--val-size", type=int, default=20)
    ap.add_argument("--val-max-steps", type=int, default=1500)
    ap.add_argument("--val-final-window-frac", type=float, default=0.2)
    ap.add_argument("--val-epsilon-beta", type=float, default=0.10)
    ap.add_argument("--val-dwell-time", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=25)
    ap.add_argument("--min-episodes", type=int, default=200)
    ap.add_argument("--patience-evals", type=int, default=6)
    ap.add_argument("--delta-sr", type=float, default=0.05)
    ap.add_argument("--delta-err", type=float, default=0.02)
    ap.add_argument("--best-score-alpha", type=float, default=0.25)
    ap.add_argument("--score-min-delta", type=float, default=1e-4)

    ap.add_argument("--train-seeds", type=str, default="0,1,2")
    ap.add_argument("--reset-seed", type=int, default=None)
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    ap.add_argument("--resume", action="store_true", help="Skip layouts already marked successful.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--run-ids", type=str, default=None, help="Optional comma list of layout names to run.")
    return ap.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_csv_str(expr: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in expr.split(","):
        tok = raw.strip()
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def parse_seed_csv(expr: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for raw in expr.split(","):
        tok = raw.strip()
        if not tok:
            continue
        seed = int(tok)
        if seed not in seen:
            seen.add(seed)
            out.append(seed)
    if not out:
        raise ValueError("train-seeds must contain at least one integer seed.")
    return out


def list_numeric_actor_episodes(model_dir: Path) -> List[int]:
    episodes: List[int] = []
    if not model_dir.exists():
        return episodes
    for p in model_dir.iterdir():
        name = p.name
        if not name.endswith(NUMERIC_ACTOR_SUFFIX):
            continue
        stem = name[: -len(NUMERIC_ACTOR_SUFFIX)]
        if stem.isdigit():
            episodes.append(int(stem))
    return sorted(episodes)


def pick_selected_episode(model_dir: Path) -> Optional[int]:
    best_meta = model_dir / "best_checkpoint_meta.json"
    if best_meta.exists():
        try:
            payload = json.loads(best_meta.read_text(encoding="utf-8"))
            episode = payload.get("episode")
            if episode is not None:
                episode = int(episode)
                if (model_dir / f"{episode}_actor.pt").exists():
                    return episode
        except Exception:
            pass
    episodes = list_numeric_actor_episodes(model_dir)
    return episodes[-1] if episodes else None


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"placements manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    layouts = payload.get("layouts")
    if not isinstance(layouts, list) or not layouts:
        raise ValueError("placements manifest must contain a non-empty 'layouts' list.")
    for layout in layouts:
        for req in ["name", "sensor_indices"]:
            if req not in layout:
                raise ValueError(f"layout entry missing required field '{req}': {layout}")
    return layouts


def build_train_command(
    args: argparse.Namespace,
    train_script: Path,
    run_name: str,
    run_root: Path,
    model_dir: Path,
    buffer_dir: Path,
    sensor_indices_file: Path,
    state_dim: int,
    train_seed: int,
    reset_seed: int,
    setup_name: str,
) -> List[str]:
    cmd = [
        args.python_bin,
        str(train_script),
        "--setup-name",
        setup_name,
        "--exp-name",
        run_name,
        "--model-dir",
        str(model_dir),
        "--buffer-dir",
        str(buffer_dir),
        "--run-dir",
        str(run_root),
        "--target-file",
        args.target_file,
        "--x-file",
        args.x_file,
        "--init-file",
        args.init_file,
        "--init-split-file",
        args.init_split_file,
        "--split-seed",
        str(args.split_seed),
        "--train-split-size",
        str(args.train_split_size),
        "--val-split-size",
        str(args.val_split_size),
        "--test-split-size",
        str(args.test_split_size),
        "--max-episodes",
        str(args.max_episodes),
        "--max-steps",
        str(args.max_steps),
        "--max-total-reward",
        str(args.max_total_reward),
        "--state-dim",
        str(state_dim),
        "--action-dim",
        str(args.action_dim),
        "--action-lim",
        str(args.action_lim),
        "--domain-length",
        str(args.domain_length),
        "--save-every",
        str(args.save_every),
        "--sensor-indices-file",
        str(sensor_indices_file),
        "--train-seed",
        str(train_seed),
        "--reset-seed",
        str(reset_seed),
        "--val-seed",
        str(args.val_seed),
        "--val-size",
        str(args.val_size),
        "--val-max-steps",
        str(args.val_max_steps),
        "--val-final-window-frac",
        str(args.val_final_window_frac),
        "--val-epsilon-beta",
        str(args.val_epsilon_beta),
        "--val-dwell-time",
        str(args.val_dwell_time),
        "--eval-interval",
        str(args.eval_interval),
        "--min-episodes",
        str(args.min_episodes),
        "--patience-evals",
        str(args.patience_evals),
        "--delta-sr",
        str(args.delta_sr),
        "--delta-err",
        str(args.delta_err),
        "--best-score-alpha",
        str(args.best_score_alpha),
        "--score-min-delta",
        str(args.score_min_delta),
        "--device",
        args.device,
    ]
    if args.early_stop_enabled:
        cmd.append("--early-stop-enabled")
    else:
        cmd.append("--no-early-stop")
    return cmd


def main() -> None:
    args = parse_args()
    if Path(args.init_file).name != "INIT.dat":
        raise ValueError(f"init-file must point to INIT.dat under the study protocol, got '{Path(args.init_file).name}'.")
    manifest_path = Path(args.placements_manifest)
    bundle_root = Path(args.bundle_root)
    train_script = Path(args.train_script)
    layouts = load_manifest(manifest_path)
    train_seeds = parse_seed_csv(args.train_seeds)
    selected_run_ids = set(parse_csv_str(args.run_ids)) if args.run_ids else None

    runs_dir = bundle_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    campaign_runs: List[Dict[str, Any]] = []
    target_name = Path(args.target_file).stem
    halted = False

    for layout in layouts:
        if halted:
            break
        layout_name = str(layout["name"])
        if selected_run_ids is not None and layout_name not in selected_run_ids:
            continue

        sensor_indices = [int(v) for v in layout["sensor_indices"]]
        state_dim = len(sensor_indices)
        setup_name = str(args.setup_name or layout_name)

        for train_seed in train_seeds:
            reset_seed = int(args.reset_seed if args.reset_seed is not None else train_seed)
            run_id = f"{layout_name}__{target_name}__seed_{train_seed}"
            run_root = runs_dir / setup_name / layout_name / target_name / f"seed_{train_seed}"
            model_dir = run_root / "model"
            buffer_dir = run_root / "buffer"
            sensor_indices_file = run_root / "sensor_indices.json"
            run_status_path = run_root / "run_status.json"
            train_log_path = run_root / "train.log"

            run_root.mkdir(parents=True, exist_ok=True)
            write_json(sensor_indices_file, {"sensor_indices": sensor_indices})

            record: Dict[str, Any] = {
                "run_id": run_id,
                "setup_name": setup_name,
                "layout_name": layout_name,
                "target_name": target_name,
                "rl_seed": int(train_seed),
                "reset_seed": int(reset_seed),
                "k": state_dim,
                "strategy": layout.get("strategy", "custom"),
                "source": "trained",
                "role": "other",
                "run_root": str(run_root),
                "model_dir": str(model_dir),
                "buffer_dir": str(buffer_dir),
                "sensor_indices": sensor_indices,
                "status": "pending",
            }

            if args.resume and run_status_path.exists():
                try:
                    status_payload = json.loads(run_status_path.read_text(encoding="utf-8"))
                    if status_payload.get("status") == "success":
                        record["status"] = "success"
                        record["updated_at"] = status_payload.get("updated_at")
                        record["selected_episode"] = status_payload.get("selected_episode")
                        campaign_runs.append(record)
                        continue
                except Exception:
                    pass

            cmd = build_train_command(
                args=args,
                train_script=train_script,
                run_name=run_id,
                run_root=run_root,
                model_dir=model_dir,
                buffer_dir=buffer_dir,
                sensor_indices_file=sensor_indices_file,
                state_dim=state_dim,
                train_seed=int(train_seed),
                reset_seed=int(reset_seed),
                setup_name=setup_name,
            )
            record["command"] = cmd

            if args.dry_run:
                record["status"] = "dry_run"
                campaign_runs.append(record)
                continue

            with train_log_path.open("w", encoding="utf-8") as logf:
                proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT))
            record["status"] = "success" if proc.returncode == 0 else "failed"
            record["updated_at"] = now_iso()
            record["returncode"] = int(proc.returncode)
            record["train_log"] = str(train_log_path)

            if record["status"] == "success":
                record["selected_episode"] = pick_selected_episode(model_dir)

            campaign_runs.append(record)
            if proc.returncode != 0 and args.stop_on_fail:
                halted = True
                break

    manifest = {
        "generated_at": now_iso(),
        "placements_manifest": str(manifest_path),
        "bundle_root": str(bundle_root),
        "target_file": args.target_file,
        "init_file": args.init_file,
        "init_split_file": args.init_split_file,
        "split_seed": int(args.split_seed),
        "train_seeds": train_seeds,
        "total_runs": len(campaign_runs),
        "runs": campaign_runs,
    }
    write_json(bundle_root / "campaign_manifest.json", manifest)

    models = []
    for run in campaign_runs:
        if run.get("status") != "success":
            continue
        models.append(
            {
                "name": run["run_id"],
                "role": run["role"],
                "setup_name": run["setup_name"],
                "target_name": run["target_name"],
                "rl_seed": run["rl_seed"],
                "target_file": args.target_file,
                "models_dir": run["model_dir"],
                "s_dim": run["k"],
                "a_dim": args.action_dim,
                "a_max": args.action_lim,
                "sensor_indices": run["sensor_indices"],
                "episode": run.get("selected_episode"),
            }
        )
    write_json(bundle_root / "models_spec_for_evaluation.json", {"models": models})


if __name__ == "__main__":
    main()
