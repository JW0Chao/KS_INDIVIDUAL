from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def build_forbidden_indices(
    actuator_indices: Sequence[int],
    state_dim: int,
    window: int,
    periodic: bool,
) -> List[int]:
    if window < 0:
        raise ValueError("exclude window must be >= 0")
    excluded = set()
    for actuator_index in actuator_indices:
        for delta in range(-window, window + 1):
            idx = actuator_index + delta
            if periodic:
                idx %= state_dim
            elif idx < 0 or idx >= state_dim:
                continue
            excluded.add(int(idx))
    return sorted(excluded)


def corridor_ranges(actuator_indices: Sequence[int], state_dim: int, window: int) -> List[Tuple[int, int]]:
    act = sorted(int(a) for a in actuator_indices)
    ranges: List[Tuple[int, int]] = []
    m = len(act)
    for i in range(m):
        a = act[i]
        nxt = act[(i + 1) % m]
        if nxt <= a:
            nxt += state_dim
        safe_start = a + window + 1
        safe_end = nxt - window - 1
        if safe_start > safe_end:
            raise ValueError(
                f"No safe corridor between actuators {a} and {nxt % state_dim} for window={window}"
            )
        ranges.append((safe_start, safe_end))
    return ranges


def interior_anchor_range(safe_start: int, safe_end: int) -> Tuple[int, int]:
    anchor_start = safe_start + 1
    anchor_end = safe_end - 1
    if anchor_start > anchor_end:
        raise ValueError(f"Anchor interval empty for safe corridor [{safe_start},{safe_end}]")
    return anchor_start, anchor_end


def evenly_spaced_inclusive(start: int, end: int, n: int) -> List[int]:
    if n <= 0:
        raise ValueError("n must be positive")
    if start > end:
        raise ValueError(f"Invalid interval [{start},{end}]")
    if n == 1:
        return [int((start + end) // 2)]
    span = end - start
    vals = [int(round(start + (span * k) / (n - 1))) for k in range(n)]
    out: List[int] = []
    seen = set()
    for val in vals:
        if val not in seen:
            seen.add(val)
            out.append(val)
    if len(out) != n:
        raise ValueError(
            f"Could not place {n} unique points in interval [{start},{end}] with inclusive spacing."
        )
    return out


def build_uniform_indices(
    k: int,
    corridor_safe_ranges: Sequence[Tuple[int, int]],
    state_dim: int,
) -> List[int]:
    if k % len(corridor_safe_ranges) != 0:
        raise ValueError(f"K={k} not divisible by corridor count={len(corridor_safe_ranges)}")
    per_corridor = k // len(corridor_safe_ranges)
    indices: List[int] = []
    for safe_start, safe_end in corridor_safe_ranges:
        anchor_start, anchor_end = interior_anchor_range(safe_start, safe_end)
        pts = evenly_spaced_inclusive(anchor_start, anchor_end, per_corridor)
        indices.extend([p % state_dim for p in pts])
    return sorted(indices)


def validate_indices(indices: Sequence[int], k: int, forbidden: set[int], state_dim: int) -> None:
    if len(indices) != k:
        raise ValueError(f"Expected {k} sensors, got {len(indices)}")
    if len(set(indices)) != len(indices):
        raise ValueError("Sensor indices contain duplicates")
    out_of_bounds = [idx for idx in indices if idx < 0 or idx >= state_dim]
    if out_of_bounds:
        raise ValueError(f"Sensor indices out of bounds: {out_of_bounds}")
    forbidden_hits = [idx for idx in indices if idx in forbidden]
    if forbidden_hits:
        raise ValueError(f"Sensor indices overlap forbidden actuator windows: {forbidden_hits}")
