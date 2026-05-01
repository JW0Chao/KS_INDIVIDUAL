from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def pick_checkpoint_episode(models_dir: Path, requested_episode: Optional[int], need_critic: bool) -> int:
    if requested_episode is not None:
        actor_path = models_dir / f"{requested_episode}_actor.pt"
        if not actor_path.exists():
            raise FileNotFoundError(f"Missing actor checkpoint: {actor_path}")
        if need_critic:
            critic_path = models_dir / f"{requested_episode}_critic.pt"
            if not critic_path.exists():
                raise FileNotFoundError(f"Missing critic checkpoint: {critic_path}")
        return requested_episode

    actor_ckpts = []
    for p in models_dir.iterdir():
        name = p.name
        if not name.endswith("_actor.pt") or name.endswith("_target_actor.pt"):
            continue
        stem = name[: -len("_actor.pt")]
        if stem.isdigit():
            actor_ckpts.append(name)
    if not actor_ckpts:
        raise FileNotFoundError(f"No actor checkpoints found in {models_dir}")
    episode = max(int(name.split("_")[0]) for name in actor_ckpts)
    if need_critic:
        critic_path = models_dir / f"{episode}_critic.pt"
        if not critic_path.exists():
            raise FileNotFoundError(f"Missing critic checkpoint: {critic_path}")
    return episode


def resolve_sensor_indices(spec: Dict[str, Any], state_dim: int) -> np.ndarray:
    s_dim = int(spec["s_dim"])
    raw = spec.get("sensor_indices")
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
        if np.all(error_curve[t : t + dwell_steps] <= epsilon):
            return t
    return None
