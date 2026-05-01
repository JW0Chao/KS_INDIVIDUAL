
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
DEFAULT_RUNS_DIR = ROOT / "artifacts" / "raw" / "runs"
STUDY_PROTOCOL_DIR = ROOT / "studies" / "controller_protocol"
DEFAULT_INIT_SPLIT_FILE = STUDY_PROTOCOL_DIR / "manifests" / "controller_setup_split.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ks_control.ks import KS
from ks_control.replay_buffer import MemoryBuffer
from ks_control.study_protocol import (
    compute_target_relative_epsilon,
    integrated_control_effort,
    load_or_create_split_manifest,
    validation_metrics_better,
)
from ks_control.training import Trainer


DEFAULT_SENSOR_INDICES_8 = [4, 12, 20, 28, 36, 44, 52, 60]
NUMERIC_ACTOR_RE = re.compile(r"^(\d+)_actor\.pt$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Train one RL controller for a specific sensor layout with reproducible run metadata and "
            "hardened convergence-state persistence."
        )
    )

    ap.add_argument("--exp-name", type=str, default="uniform_k8")
    ap.add_argument("--model-dir", type=str, default=None)
    ap.add_argument("--buffer-dir", type=str, default=None)
    ap.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory to store run-level metadata (run_config/run_status/early_stop_state). Default: model-dir.",
    )

    ap.add_argument("--setup-name", type=str, default=None)
    ap.add_argument("--target-file", type=str, default=str(DATA_DIR / "u3.dat"))
    ap.add_argument("--x-file", type=str, default=str(DATA_DIR / "x.dat"))
    ap.add_argument("--init-file", type=str, default=str(DATA_DIR / "INIT.dat"))
    ap.add_argument("--init-split-file", type=str, default=str(DEFAULT_INIT_SPLIT_FILE))
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument("--train-split-size", type=int, default=20)
    ap.add_argument("--val-split-size", type=int, default=20)
    ap.add_argument("--test-split-size", type=int, default=30)

    ap.add_argument("--max-episodes", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--max-total-reward", type=float, default=-35.0)
    ap.add_argument("--state-dim", type=int, default=8, help="Number of sensors.")
    ap.add_argument("--action-dim", type=int, default=4, help="Number of actuators.")
    ap.add_argument("--action-lim", type=float, default=1.0)
    ap.add_argument("--domain-length", type=float, default=22.0)
    ap.add_argument("--save-every", type=int, default=10, help="Periodic checkpoint frequency (episodes).")

    ap.add_argument(
        "--sensor-indices",
        type=str,
        default=None,
        help="Optional explicit indices/ranges, e.g. '4,12,20-23'.",
    )
    ap.add_argument(
        "--sensor-indices-file",
        type=str,
        default=None,
        help="Optional JSON file containing sensor indices list or {'sensor_indices': [...]}.",
    )

    early_group = ap.add_mutually_exclusive_group()
    early_group.add_argument("--early-stop-enabled", dest="early_stop_enabled", action="store_true")
    early_group.add_argument("--no-early-stop", dest="early_stop_enabled", action="store_false")
    ap.set_defaults(early_stop_enabled=True)

    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--reset-seed", type=int, default=None)
    ap.add_argument("--val-seed", type=int, default=0)
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

    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--ini", type=int, default=0, help="Start episode for restart/test loading.")

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # Deprecated protocol-breaking flags kept only for a softer transition.
    ap.add_argument("--train-init-file", dest="legacy_train_init_file", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--val-init-file", dest="legacy_val_init_file", type=str, default=None, help=argparse.SUPPRESS)

    return ap.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_index_expr(expr: str) -> List[int]:
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
                raise ValueError(f"Invalid range '{tok}': end < start.")
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
        raise ValueError("No valid indices parsed from --sensor-indices.")
    return out


def load_sensor_indices_from_file(path: Path) -> List[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        arr = payload
    elif isinstance(payload, dict):
        if "sensor_indices" in payload:
            arr = payload["sensor_indices"]
        elif "indices" in payload:
            arr = payload["indices"]
        else:
            raise ValueError(f"{path} must contain 'sensor_indices' or 'indices' key.")
    else:
        raise ValueError(f"Unsupported JSON format in {path}.")
    if not isinstance(arr, list) or len(arr) == 0:
        raise ValueError(f"{path} sensor list must be a non-empty list.")
    return [int(v) for v in arr]


def resolve_sensor_indices(args: argparse.Namespace, x_size: int) -> np.ndarray:
    if args.sensor_indices_file is not None:
        idx = np.asarray(load_sensor_indices_from_file(Path(args.sensor_indices_file)), dtype=np.int64)
    elif args.sensor_indices is not None:
        idx = np.asarray(parse_index_expr(args.sensor_indices), dtype=np.int64)
    elif args.state_dim == 8:
        idx = np.asarray(DEFAULT_SENSOR_INDICES_8, dtype=np.int64)
    else:
        step = x_size // args.state_dim
        if step <= 0:
            raise ValueError(f"Invalid sensor spacing for x_size={x_size}, state_dim={args.state_dim}.")
        idx = np.arange(0, x_size, step, dtype=np.int64)

    if idx.size != args.state_dim:
        raise ValueError(
            f"sensor_indices size mismatch: got {idx.size}, expected state_dim={args.state_dim}."
        )
    if np.unique(idx).size != idx.size:
        raise ValueError("sensor_indices contains duplicates.")
    if np.any(idx < 0) or np.any(idx >= x_size):
        raise ValueError(f"sensor_indices contains out-of-bounds index for x_size={x_size}.")
    return idx


def first_stable_step(error_curve: np.ndarray, epsilon: float, dwell_steps: int) -> Optional[int]:
    n = error_curve.shape[0]
    if dwell_steps <= 1:
        hits = np.where(error_curve <= epsilon)[0]
        return int(hits[0]) if hits.size > 0 else None
    if dwell_steps > n:
        return None
    for t in range(0, n - dwell_steps + 1):
        if np.all(error_curve[t : t + dwell_steps] <= epsilon):
            return t
    return None


def list_numeric_actor_episodes(model_dir: Path) -> List[int]:
    episodes: List[int] = []
    if not model_dir.exists():
        return episodes
    for p in model_dir.iterdir():
        m = NUMERIC_ACTOR_RE.match(p.name)
        if m:
            episodes.append(int(m.group(1)))
    return sorted(episodes)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.legacy_train_init_file is not None:
        raise ValueError(
            "--train-init-file is no longer supported. Training initial conditions must come from INIT.dat "
            "through --init-file and --init-split-file."
        )
    if args.legacy_val_init_file is not None:
        raise ValueError(
            "--val-init-file is no longer supported. Validation initial conditions must come from INIT.dat "
            "through --init-file and --init-split-file."
        )
    if Path(args.init_file).name != "INIT.dat":
        raise ValueError(
            f"init-file must point to INIT.dat under the study protocol, got '{Path(args.init_file).name}'."
        )
    if args.val_size != args.val_split_size:
        raise ValueError(
            f"val-size ({args.val_size}) must match val-split-size ({args.val_split_size}) under the study protocol."
        )

    setup_name = str(args.setup_name or args.exp_name)
    target_name = Path(args.target_file).stem
    reset_seed = int(args.train_seed if args.reset_seed is None else args.reset_seed)

    default_run_dir = DEFAULT_RUNS_DIR / args.exp_name
    model_dir = Path(args.model_dir) if args.model_dir else default_run_dir / f"Model_{args.exp_name}"
    buffer_dir = Path(args.buffer_dir) if args.buffer_dir else default_run_dir / f"Buffer_{args.exp_name}"
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    buffer_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_status_path = run_dir / "run_status.json"
    run_config_path = run_dir / "run_config.json"
    early_stop_state_path = run_dir / "early_stop_state.json"

    device = resolve_device(args.device)

    u_target = np.loadtxt(args.target_file)
    x = np.loadtxt(args.x_file)
    if x.ndim != 1:
        raise ValueError(f"x-file must be 1D, got shape={x.shape}")
    if u_target.ndim != 1 or u_target.shape[0] != x.shape[0]:
        raise ValueError(f"target-file shape must be ({x.shape[0]},), got {u_target.shape}")
    u_target = np.float32(u_target)

    sensor_indices = resolve_sensor_indices(args, x_size=x.shape[0])

    # Reproducibility setup.
    np.random.seed(args.train_seed)
    random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ks = KS(L=args.domain_length, N=x.size, a_dim=args.action_dim)

    print("Device :-", device)
    print("Setup Name :-", setup_name)
    print("Target File :-", args.target_file)
    print("Init File :-", args.init_file)
    print("Init Split File :-", args.init_split_file)
    print("State Dimensions :-", args.state_dim)
    print("Action Dimensions :-", args.action_dim)
    print("Action Max :-", args.action_lim)
    print("Sensor Indices :-", sensor_indices.tolist())
    print("Sensor Positions :-", x[sensor_indices].tolist())
    print("RL Seed :-", args.train_seed)
    print("Reset Seed :-", reset_seed)
    print("Split Seed :-", args.split_seed)
    print("Model Dir :-", str(model_dir))
    print("Buffer Dir :-", str(buffer_dir))
    print("Run Dir :-", str(run_dir))

    ram = MemoryBuffer(buffer_dir=str(buffer_dir))
    trainer = Trainer(
        args.state_dim,
        args.action_dim,
        args.action_lim,
        ram,
        device,
        args.test,
        model_dir=str(model_dir),
        buffer_dir=str(buffer_dir),
    )

    init_states = np.loadtxt(args.init_file)
    if init_states.ndim == 1:
        init_states = init_states.reshape(1, -1)
    if init_states.shape[1] != x.size:
        raise ValueError(f"{args.init_file} row length ({init_states.shape[1]}) must match x size ({x.size})")

    split_manifest = load_or_create_split_manifest(
        path=Path(args.init_split_file),
        init_file=args.init_file,
        num_rows=int(init_states.shape[0]),
        split_seed=int(args.split_seed),
        train_size=int(args.train_split_size),
        val_size=int(args.val_split_size),
        test_size=int(args.test_split_size),
    )
    train_rows = np.asarray(split_manifest["train_rows"], dtype=np.int64)
    validation_rows = np.asarray(split_manifest["val_rows"], dtype=np.int64)
    test_rows = np.asarray(split_manifest["test_rows"], dtype=np.int64)
    reset_rng = np.random.default_rng(reset_seed)

    current_ep: Optional[int] = None
    context: Dict[str, Optional[np.ndarray]] = {
        "last_observation": None,
        "last_new_observation": None,
    }
    def _save_training_snapshot(ep: int) -> None:
        trainer.save_models(ep)
        if context["last_observation"] is not None:
            np.savetxt(buffer_dir / "state.dat", context["last_observation"])
        if hasattr(ks, "f0"):
            np.savetxt(buffer_dir / "action.dat", ks.f0)
        if context["last_new_observation"] is not None:
            np.savetxt(buffer_dir / "new_state.dat", context["last_new_observation"])

    def _copy_best_checkpoint_aliases(ep: int) -> None:
        ckpt_map = {
            f"{ep}_actor.pt": "best_actor.pt",
            f"{ep}_critic.pt": "best_critic.pt",
            f"{ep}_target_actor.pt": "best_target_actor.pt",
            f"{ep}_target_critic.pt": "best_target_critic.pt",
        }
        for src_name, dst_name in ckpt_map.items():
            src = model_dir / src_name
            dst = model_dir / dst_name
            if not src.exists():
                raise FileNotFoundError(f"Expected checkpoint missing for best alias copy: {src}")
            shutil.copy2(src, dst)

    def _append_validation_history_row(path: Path, row: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                f"{row['episode']},{row['success_rate']:.6f},{row['mean_final_error']:.6f},"
                f"{row['mean_control_effort']:.6f},{int(row['is_best'])},"
                f"{row['no_improve_eval_count']},{row['best_episode']}\n"
            )

    def _update_run_status(status: str, **kwargs: Any) -> None:
        payload: Dict[str, Any] = {
            "status": status,
            "updated_at": now_iso(),
            "exp_name": args.exp_name,
            "setup_name": setup_name,
            "target_name": target_name,
            "target_file": args.target_file,
            "model_dir": str(model_dir),
            "buffer_dir": str(buffer_dir),
            "run_dir": str(run_dir),
            "test_mode": bool(args.test),
            "rl_seed": int(args.train_seed),
            "reset_seed": int(reset_seed),
            "split_seed": int(args.split_seed),
        }
        payload.update(kwargs)
        write_json(run_status_path, payload)

    # Persisted paths for convergence reproducibility.
    validation_rows_path = model_dir / "validation_rows.json"
    validation_history_path = model_dir / "validation_history.csv"
    best_meta_path = model_dir / "best_checkpoint_meta.json"

    # State variables for early stop.
    best_episode = -1
    best_validation: Optional[Dict[str, Any]] = None
    no_improve_eval_count = 0
    early_stop_triggered = False
    early_stop_reason = ""
    last_eval_episode = -1

    def _save_early_stop_state() -> None:
        payload = {
            "best_episode": int(best_episode),
            "best_validation": best_validation,
            "no_improve_eval_count": int(no_improve_eval_count),
            "early_stop_triggered": bool(early_stop_triggered),
            "early_stop_reason": str(early_stop_reason),
            "last_eval_episode": int(last_eval_episode),
        }
        write_json(early_stop_state_path, payload)

    def _load_early_stop_state_if_present() -> None:
        nonlocal best_episode
        nonlocal best_validation
        nonlocal no_improve_eval_count
        nonlocal early_stop_triggered
        nonlocal early_stop_reason
        nonlocal last_eval_episode
        if not early_stop_state_path.exists():
            return
        payload = json.loads(early_stop_state_path.read_text(encoding="utf-8"))
        best_episode = int(payload.get("best_episode", -1))
        best_validation = payload.get("best_validation", None)
        no_improve_eval_count = int(payload.get("no_improve_eval_count", 0))
        early_stop_triggered = bool(payload.get("early_stop_triggered", False))
        early_stop_reason = str(payload.get("early_stop_reason", ""))
        last_eval_episode = int(payload.get("last_eval_episode", -1))

    def _ensure_best_aliases() -> None:
        best_aliases = [
            model_dir / "best_actor.pt",
            model_dir / "best_critic.pt",
            model_dir / "best_target_actor.pt",
            model_dir / "best_target_critic.pt",
        ]
        if all(p.exists() for p in best_aliases):
            return
        if best_episode >= 0:
            _copy_best_checkpoint_aliases(best_episode)
            return
        episodes = list_numeric_actor_episodes(model_dir)
        if not episodes:
            return
        fallback_ep = episodes[-1]
        _copy_best_checkpoint_aliases(fallback_ep)
        fallback_meta = {
            "episode": int(fallback_ep),
            "note": "No validation improvement recorded; best aliases set to latest numeric checkpoint.",
            "stop_metric": "validation_lexicographic",
            "rl_seed": int(args.train_seed),
            "reset_seed": int(reset_seed),
            "split_seed": int(args.split_seed),
        }
        if not best_meta_path.exists():
            write_json(best_meta_path, fallback_meta)

    def run_validation_eval() -> Dict[str, Any]:
        ks_val = KS(L=args.domain_length, N=x.size, a_dim=args.action_dim)
        dwell_steps = int(np.ceil(args.val_dwell_time / ks_val.dt))
        if dwell_steps <= 0:
            raise ValueError(
                f"Invalid dwell steps computed from val_dwell_time={args.val_dwell_time} and dt={ks_val.dt}"
            )
        if not (0.0 < args.val_final_window_frac <= 1.0):
            raise ValueError(
                f"val_final_window_frac must be in (0,1], got {args.val_final_window_frac}"
            )
        final_window = max(1, int(np.ceil(args.val_final_window_frac * args.val_max_steps)))
        final_start = args.val_max_steps - final_window
        epsilon = compute_target_relative_epsilon(u_target=u_target, epsilon_beta=args.val_epsilon_beta)

        successes = 0
        final_errors = np.zeros(validation_rows.size, dtype=np.float32)
        control_efforts = np.zeros(validation_rows.size, dtype=np.float32)
        with torch.no_grad():
            for k, row_idx in enumerate(validation_rows):
                obs = np.float32(init_states[int(row_idx)].copy())
                error_curve = np.zeros(args.val_max_steps, dtype=np.float32)
                action_curve = np.zeros((args.val_max_steps, args.action_dim), dtype=np.float32)
                for t in range(args.val_max_steps):
                    state = np.float32(obs[sensor_indices])
                    action = trainer.get_action(state, Test=True)
                    action_curve[t] = np.float32(action)
                    obs = ks_val.advance(obs, action)
                    error_curve[t] = np.float32(np.linalg.norm(obs - u_target))
                final_errors[k] = np.float32(np.mean(error_curve[final_start:]))
                control_efforts[k] = np.float32(integrated_control_effort(action_curve, dt=float(ks_val.dt)))
                if first_stable_step(error_curve, epsilon=epsilon, dwell_steps=dwell_steps) is not None:
                    successes += 1
        return {
            "success_rate": float(successes / validation_rows.size),
            "mean_final_error": float(np.mean(final_errors)),
            "mean_control_effort": float(np.mean(control_efforts)),
        }

    # Write initial run configuration.
    run_cfg: Dict[str, Any] = {
        "created_at": now_iso(),
        "argv": sys.argv,
        "exp_name": args.exp_name,
        "setup_name": setup_name,
        "target_name": target_name,
        "model_dir": str(model_dir),
        "buffer_dir": str(buffer_dir),
        "run_dir": str(run_dir),
        "device": str(device),
        "target_file": args.target_file,
        "x_file": args.x_file,
        "init_file": args.init_file,
        "init_split_file": args.init_split_file,
        "split_seed": int(args.split_seed),
        "train_split_size": int(args.train_split_size),
        "val_split_size": int(args.val_split_size),
        "test_split_size": int(args.test_split_size),
        "train_rows": train_rows.tolist(),
        "val_rows": validation_rows.tolist(),
        "test_rows": test_rows.tolist(),
        "max_episodes": int(args.max_episodes),
        "max_steps": int(args.max_steps),
        "max_total_reward": float(args.max_total_reward),
        "state_dim": int(args.state_dim),
        "action_dim": int(args.action_dim),
        "action_lim": float(args.action_lim),
        "domain_length": float(args.domain_length),
        "save_every": int(args.save_every),
        "sensor_indices": sensor_indices.tolist(),
        "train_seed": int(args.train_seed),
        "rl_seed": int(args.train_seed),
        "reset_seed": int(reset_seed),
        "val_seed": int(args.val_seed),
        "early_stop_enabled": bool(args.early_stop_enabled),
        "val_size": int(args.val_size),
        "val_max_steps": int(args.val_max_steps),
        "val_final_window_frac": float(args.val_final_window_frac),
        "val_epsilon_beta": float(args.val_epsilon_beta),
        "val_epsilon_mode": "target_relative",
        "val_dwell_time": float(args.val_dwell_time),
        "eval_interval": int(args.eval_interval),
        "min_episodes": int(args.min_episodes),
        "patience_evals": int(args.patience_evals),
        "checkpoint_rule": [
            "highest validation success rate",
            "tie-break by lower mean final error",
            "tie-break by lower control effort",
        ],
        "restart": bool(args.restart),
        "test": bool(args.test),
        "plot": bool(args.plot),
        "ini": int(args.ini),
    }
    write_json(run_config_path, run_cfg)
    _update_run_status("pending", message="Configuration written.")
    if args.restart or args.test:
        ini = int(args.ini)
        if args.test and not (model_dir / f"{ini}_actor.pt").exists():
            actor_ckpts = list_numeric_actor_episodes(model_dir)
            if not actor_ckpts:
                raise FileNotFoundError(f"No actor checkpoints found in {model_dir}")
            ini = actor_ckpts[-1]
            print("Test checkpoint not found for ini; using latest episode:", ini)
        trainer.load_models(ini, args.test)
    else:
        ini = int(args.ini)

    # Early-stop/validation setup.
    if args.early_stop_enabled and not args.test:
        if args.val_max_steps <= 0:
            raise ValueError(f"val_max_steps must be positive, got {args.val_max_steps}")
        if args.eval_interval <= 0:
            raise ValueError(f"eval_interval must be positive, got {args.eval_interval}")
        if args.patience_evals <= 0:
            raise ValueError(f"patience_evals must be positive, got {args.patience_evals}")
        if validation_rows.size != args.val_split_size:
            raise ValueError(
                f"val_rows size ({validation_rows.size}) must match val-split-size ({args.val_split_size})."
            )

        validation_rows_source = "shared_split_manifest"
        if validation_rows_path.exists():
            payload = json.loads(validation_rows_path.read_text(encoding="utf-8"))
            if payload.get("rows") != validation_rows.tolist():
                raise ValueError(f"{validation_rows_path} rows do not match the shared split manifest.")
        else:
            validation_rows_payload = {
                "source": "shared_split_manifest",
                "split_file": args.init_split_file,
                "split_seed": int(args.split_seed),
                "init_file": args.init_file,
                "val_size": int(validation_rows.size),
                "role": "val",
                "rows": validation_rows.tolist(),
                "stop_metric": "validation_lexicographic",
            }
            write_json(validation_rows_path, validation_rows_payload)

        if not validation_history_path.exists():
            with open(validation_history_path, "w", encoding="utf-8") as f:
                f.write(
                    "episode,success_rate,mean_final_error,mean_control_effort,"
                    "is_best,no_improve_eval_count,best_episode\n"
                )

        if args.restart:
            _load_early_stop_state_if_present()

        print(
            "Validation setup complete:",
            "size=", validation_rows.size,
            "split_file=", str(args.init_split_file),
            "rows_file=", str(validation_rows_path),
            "source=", validation_rows_source,
        )
        _save_early_stop_state()
    else:
        # Keep a state artifact for auditability even when early-stop is disabled/test mode.
        _save_early_stop_state()

    _update_run_status("running", message="Training loop started.", ini=int(ini))

    interrupted = False
    failure: Optional[Exception] = None

    try:
        for _ep in range(ini, args.max_episodes):
            current_ep = _ep
            episode_rows = test_rows if args.test else train_rows
            if episode_rows.size == 0:
                raise ValueError("Episode reset row set is empty.")
            init_idx = int(reset_rng.choice(episode_rows))
            new_observation = np.float32(init_states[init_idx].copy())

            for r in range(args.max_steps):
                state = np.float32(new_observation[sensor_indices])
                observation = new_observation
                action = trainer.get_action(state, Test=args.test)
                new_observation = ks.advance(observation, action)
                reward = -np.linalg.norm(new_observation - u_target)
                new_state = np.float32(new_observation[sensor_indices])

                context["last_observation"] = observation
                context["last_new_observation"] = new_observation

                if reward < args.max_total_reward and not args.test:
                    reward = -100
                    trainer.ram.add(state, action, reward, new_state, args.test)
                    break
                trainer.ram.add(state, action, reward, new_state, args.test)

                trainer.optimize(args.test)
                if (r % 20 == 0) and args.plot:
                    plt.clf()
                    plt.plot(x, u_target)
                    plt.plot(x, new_observation)
                    plt.plot(x[sensor_indices], new_observation[sensor_indices], "o")
                    plt.pause(0.05)
                    plt.show(block=False)

            trainer.update_pert(args.test)
            gc.collect()

            print(
                "EPISODE :-",
                _ep,
                "init_row:",
                init_idx,
                "rew:",
                np.float32(reward),
                "memory:",
                np.float32(trainer.ram.len / trainer.ram.maxSize * 100),
                "%",
                "update:",
                np.float32(trainer.update),
                "c_loss:",
                np.float32(trainer.last_critic_loss),
                "a_loss:",
                np.float32(trainer.last_actor_loss),
            )

            if (_ep % args.save_every == 0) and (not args.test):
                _save_training_snapshot(_ep)

            if args.early_stop_enabled and (not args.test) and (((_ep - ini + 1) % args.eval_interval) == 0):
                validation_metrics = run_validation_eval()
                is_best = validation_metrics_better(validation_metrics, best_validation)

                if is_best:
                    _save_training_snapshot(_ep)
                    _copy_best_checkpoint_aliases(_ep)
                    best_episode = _ep
                    best_validation = dict(validation_metrics)
                    no_improve_eval_count = 0
                    best_meta = {
                        "episode": int(_ep),
                        "stop_metric": "validation_lexicographic",
                        "success_rate": float(validation_metrics["success_rate"]),
                        "mean_final_error": float(validation_metrics["mean_final_error"]),
                        "mean_control_effort": float(validation_metrics["mean_control_effort"]),
                        "eval_interval": int(args.eval_interval),
                        "min_episodes": int(args.min_episodes),
                        "patience_evals": int(args.patience_evals),
                        "val_size": int(validation_rows.size),
                        "val_max_steps": int(args.val_max_steps),
                        "val_final_window_frac": float(args.val_final_window_frac),
                        "val_epsilon_beta": float(args.val_epsilon_beta),
                        "val_epsilon_mode": "target_relative",
                        "val_dwell_time": float(args.val_dwell_time),
                        "setup_name": setup_name,
                        "target_name": target_name,
                        "target_file": args.target_file,
                        "init_file": args.init_file,
                        "init_split_file": args.init_split_file,
                        "split_seed": int(args.split_seed),
                        "rl_seed": int(args.train_seed),
                        "reset_seed": int(reset_seed),
                        "val_rows_file": str(validation_rows_path),
                    }
                    write_json(best_meta_path, best_meta)
                elif (_ep + 1) >= args.min_episodes:
                    no_improve_eval_count += 1

                row = {
                    "episode": _ep,
                    "success_rate": float(validation_metrics["success_rate"]),
                    "mean_final_error": float(validation_metrics["mean_final_error"]),
                    "mean_control_effort": float(validation_metrics["mean_control_effort"]),
                    "is_best": bool(is_best),
                    "no_improve_eval_count": no_improve_eval_count,
                    "best_episode": int(best_episode),
                }
                _append_validation_history_row(validation_history_path, row)
                last_eval_episode = _ep
                _save_early_stop_state()

                print(
                    "VALIDATION :- episode:",
                    _ep,
                    "success_rate:",
                    f"{validation_metrics['success_rate']:.3f}",
                    "mean_final_error:",
                    f"{validation_metrics['mean_final_error']:.6f}",
                    "mean_control_effort:",
                    f"{validation_metrics['mean_control_effort']:.6f}",
                    "is_best:",
                    is_best,
                    "no_improve_eval_count:",
                    no_improve_eval_count,
                )

                if (_ep + 1) >= args.min_episodes and no_improve_eval_count >= args.patience_evals:
                    early_stop_triggered = True
                    early_stop_reason = (
                        f"Early stop at episode {_ep}: no validation improvement for {args.patience_evals} evaluations "
                        f"(interval={args.eval_interval}, min_episodes={args.min_episodes})."
                    )
                    print(early_stop_reason)
                    _save_early_stop_state()
                    break

    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received. Stopping training early.")
        if (not args.test) and (current_ep is not None):
            print(f"Saving interruption checkpoint at episode {current_ep}...")
            _save_training_snapshot(current_ep)
        _save_early_stop_state()
        _update_run_status(
            "interrupted",
            message="KeyboardInterrupt",
            current_episode=None if current_ep is None else int(current_ep),
            early_stop_triggered=bool(early_stop_triggered),
            early_stop_reason=early_stop_reason,
            best_episode=int(best_episode),
            best_validation=best_validation,
        )
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        failure = exc
        if (not args.test) and (current_ep is not None):
            try:
                _save_training_snapshot(current_ep)
            except Exception:
                pass
        _save_early_stop_state()
        _update_run_status(
            "failed",
            message=f"{type(exc).__name__}: {exc}",
            current_episode=None if current_ep is None else int(current_ep),
            early_stop_triggered=bool(early_stop_triggered),
            early_stop_reason=early_stop_reason,
            best_episode=int(best_episode),
            best_validation=best_validation,
        )
        raise
    finally:
        if args.plot:
            plt.close("all")

    if failure is None and (not interrupted):
        # Ensure final snapshot always exists for the terminal episode.
        if (not args.test) and (current_ep is not None):
            _save_training_snapshot(current_ep)

        if args.early_stop_enabled and (not args.test):
            _ensure_best_aliases()
            _save_early_stop_state()

        selected_episode: Optional[int] = None
        if best_episode >= 0:
            selected_episode = int(best_episode)
        else:
            numeric_eps = list_numeric_actor_episodes(model_dir)
            if numeric_eps:
                selected_episode = int(numeric_eps[-1])

        if early_stop_triggered:
            print("Training ended by early stopping.")
            if best_episode >= 0:
                print("Best checkpoint episode:", best_episode)
                if best_validation is not None:
                    print("Best checkpoint validation:", json.dumps(best_validation, indent=2))
                print("Best checkpoint metadata:", str(best_meta_path))
        else:
            print("Completed episodes")

        _update_run_status(
            "success",
            message="Training completed.",
            current_episode=None if current_ep is None else int(current_ep),
            early_stop_triggered=bool(early_stop_triggered),
            early_stop_reason=early_stop_reason,
            best_episode=int(best_episode),
            best_validation=best_validation,
            selected_episode=selected_episode,
            validation_rows_file=str(validation_rows_path) if validation_rows_path.exists() else None,
            validation_history_file=str(validation_history_path) if validation_history_path.exists() else None,
            best_checkpoint_meta_file=str(best_meta_path) if best_meta_path.exists() else None,
        )


if __name__ == "__main__":
    main()
