from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


def _sample_std(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _normalize_init_file_marker(init_file: str) -> str:
    # The study protocol only permits INIT.dat as the initial-condition bank.
    # Comparing by basename keeps the split manifest portable across Windows/Linux paths.
    return Path(str(init_file)).name


def compute_target_relative_epsilon(u_target: Sequence[float], epsilon_beta: float) -> float:
    if not (0.0 < float(epsilon_beta) <= 1.0):
        raise ValueError("epsilon_beta must be in (0, 1].")
    target = np.asarray(u_target, dtype=np.float64)
    if target.ndim != 1 or target.size == 0:
        raise ValueError("u_target must be a non-empty 1D sequence.")
    return float(float(epsilon_beta) * np.linalg.norm(target))


def build_study_split(
    num_rows: int,
    split_seed: int,
    train_size: int = 20,
    val_size: int = 20,
    test_size: int = 30,
) -> Dict[str, Any]:
    if num_rows <= 0:
        raise ValueError("num_rows must be positive.")
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("train_size, val_size, and test_size must be positive.")
    total = int(train_size) + int(val_size) + int(test_size)
    if total > num_rows:
        raise ValueError(
            f"Requested split sizes train={train_size}, val={val_size}, test={test_size} exceed num_rows={num_rows}."
        )

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(num_rows).astype(np.int64)
    train_rows = np.sort(perm[:train_size]).tolist()
    val_rows = np.sort(perm[train_size : train_size + val_size]).tolist()
    test_rows = np.sort(perm[train_size + val_size : total]).tolist()
    return {
        "split_seed": int(split_seed),
        "num_rows": int(num_rows),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
    }


def validate_split_manifest(payload: Mapping[str, Any], init_file: str, num_rows: int) -> Dict[str, Any]:
    out = dict(payload)
    required = ["split_seed", "train_size", "val_size", "test_size", "train_rows", "val_rows", "test_rows"]
    for key in required:
        if key not in out:
            raise ValueError(f"Split manifest missing required key '{key}'.")

    expected_init = _normalize_init_file_marker(init_file)
    actual_init = _normalize_init_file_marker(str(out.get("init_file", "")))
    if actual_init != expected_init:
        raise ValueError(f"Split manifest init_file mismatch: {out.get('init_file')} != {init_file}")
    if int(out.get("num_rows", num_rows)) != int(num_rows):
        raise ValueError(f"Split manifest num_rows mismatch: {out.get('num_rows')} != {num_rows}")

    all_rows: List[int] = []
    for role, size_key in [("train", "train_size"), ("val", "val_size"), ("test", "test_size")]:
        rows = out[f"{role}_rows"]
        if not isinstance(rows, list):
            raise ValueError(f"Split manifest field '{role}_rows' must be a list.")
        if len(rows) != int(out[size_key]):
            raise ValueError(
                f"Split manifest field '{role}_rows' length {len(rows)} != declared {size_key} {out[size_key]}."
            )
        all_rows.extend(int(v) for v in rows)

    if len(set(all_rows)) != len(all_rows):
        raise ValueError("Split manifest row sets must be non-overlapping.")
    if any(row < 0 or row >= num_rows for row in all_rows):
        raise ValueError("Split manifest contains out-of-bounds row indices.")
    return out


def load_or_create_split_manifest(
    path: Path,
    init_file: str,
    num_rows: int,
    split_seed: int,
    train_size: int = 20,
    val_size: int = 20,
    test_size: int = 30,
) -> Dict[str, Any]:
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return validate_split_manifest(payload, init_file=init_file, num_rows=num_rows)

    payload = build_study_split(
        num_rows=num_rows,
        split_seed=split_seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )
    payload["init_file"] = _normalize_init_file_marker(init_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def validation_metrics_better(candidate: Mapping[str, Any], incumbent: Optional[Mapping[str, Any]]) -> bool:
    if incumbent is None:
        return True

    cand_sr = float(candidate["success_rate"])
    inc_sr = float(incumbent["success_rate"])
    if cand_sr != inc_sr:
        return cand_sr > inc_sr

    cand_err = float(candidate["mean_final_error"])
    inc_err = float(incumbent["mean_final_error"])
    if cand_err != inc_err:
        return cand_err < inc_err

    cand_effort = float(candidate["mean_control_effort"])
    inc_effort = float(incumbent["mean_control_effort"])
    return cand_effort < inc_effort


def summarize_run_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("rows must be non-empty.")

    success = np.asarray([1.0 if bool(r["success"]) else 0.0 for r in rows], dtype=np.float64)
    final_error = np.asarray([float(r["final_error"]) for r in rows], dtype=np.float64)
    integrated_error = np.asarray([float(r["integrated_error"]) for r in rows], dtype=np.float64)
    control_effort = np.asarray([float(r["control_effort"]) for r in rows], dtype=np.float64)
    tts_values = np.asarray(
        [np.nan if r.get("time_to_stabilize") is None else float(r["time_to_stabilize"]) for r in rows],
        dtype=np.float64,
    )
    success_mask = ~np.isnan(tts_values)

    return {
        "num_rows": int(len(rows)),
        "num_success": int(np.sum(success)),
        "success_rate": float(np.mean(success)),
        "mean_final_error": float(np.mean(final_error)),
        "mean_integrated_error": float(np.mean(integrated_error)),
        "mean_control_effort": float(np.mean(control_effort)),
        "mean_tts_success_only": float(np.nanmean(tts_values)) if np.any(success_mask) else None,
        "median_tts_success_only": float(np.nanmedian(tts_values)) if np.any(success_mask) else None,
    }


def summarize_setup_runs(run_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not run_summaries:
        raise ValueError("run_summaries must be non-empty.")

    def series(key: str) -> np.ndarray:
        return np.asarray([float(run[key]) for run in run_summaries], dtype=np.float64)

    out: Dict[str, Dict[str, float]] = {}
    for key in ["success_rate", "mean_final_error", "mean_integrated_error", "mean_control_effort"]:
        values = series(key)
        out[key] = {
            "mean": float(np.mean(values)),
            "std": _sample_std(values),
            "median": float(np.median(values)),
        }
    return out


def integrated_error(error_curve: np.ndarray, dt: float) -> float:
    return float(np.sum(np.asarray(error_curve, dtype=np.float64)) * float(dt))


def integrated_control_effort(action_curve: np.ndarray, dt: float) -> float:
    action_curve = np.asarray(action_curve, dtype=np.float64)
    return float(np.sum(np.sum(np.square(action_curve), axis=-1)) * float(dt))
