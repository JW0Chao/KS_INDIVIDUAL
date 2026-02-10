from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch

import buffer
import train
from KS import KS


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
                raise ValueError(f"Invalid sensor range '{tok}': end < start")
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


def _parse_sensor_indices(expr_or_path: str, state_dim: int) -> np.ndarray:
    p = Path(expr_or_path)
    if p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "indices" in data:
                data = data["indices"]
            else:
                raise ValueError("Sensor JSON dict must contain key 'indices'.")
        if not isinstance(data, list):
            raise ValueError("Sensor JSON must be a list or {'indices': [...]} .")
        vals: List[int] = []
        seen = set()
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

    idx = np.asarray(vals, dtype=np.int64)
    if np.unique(idx).size != idx.size:
        raise ValueError("sensor-indices contains duplicates")
    if np.any(idx < 0) or np.any(idx >= state_dim):
        raise ValueError(f"sensor-indices out of bounds for state_dim={state_dim}: {idx.tolist()}")
    return idx


def _build_sensor_indices(args: argparse.Namespace, x_size: int) -> np.ndarray:
    if args.sensor_indices is not None:
        sensor_indices = _parse_sensor_indices(args.sensor_indices, state_dim=x_size)
        if args.s_dim is not None and sensor_indices.size != args.s_dim:
            raise ValueError(
                f"sensor-indices size ({sensor_indices.size}) must match s-dim ({args.s_dim})"
            )
        return sensor_indices

    s_dim = args.s_dim
    if s_dim is None:
        raise ValueError("Either --s-dim or --sensor-indices must be provided.")

    if s_dim == 64:
        return np.arange(0, x_size, dtype=np.int64)
    if s_dim == 8:
        return np.asarray([4, 12, 20, 28, 36, 44, 52, 60], dtype=np.int64)

    sensor_step = x_size // s_dim
    if sensor_step <= 0:
        raise ValueError(f"Invalid sensor step: x_size={x_size}, s_dim={s_dim}")
    sensor_indices = np.arange(0, x_size, sensor_step, dtype=np.int64)
    if sensor_indices.size != s_dim:
        raise ValueError(
            f"Equispaced selection produced {sensor_indices.size} sensors, expected {s_dim}."
        )
    return sensor_indices


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train/test one DDPG controller case with configurable sensors.")
    ap.add_argument("--exp-name", type=str, default="run")
    ap.add_argument("--model-dir", type=str, default=None)
    ap.add_argument("--buffer-dir", type=str, default=None)

    ap.add_argument("--x-file", type=str, default="x.dat")
    ap.add_argument("--target-file", type=str, default="u3.dat")
    ap.add_argument("--train-init-file", type=str, default="u2.dat")
    ap.add_argument("--test-init-file", type=str, default="INIT.dat")

    ap.add_argument("--max-episodes", type=int, default=800)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--max-total-reward", type=float, default=-35.0)
    ap.add_argument("--save-every", type=int, default=10)

    ap.add_argument("--s-dim", type=int, default=None)
    ap.add_argument("--a-dim", type=int, default=4)
    ap.add_argument("--a-max", type=float, default=0.5)
    ap.add_argument("--domain-length", type=float, default=22.0)
    ap.add_argument(
        "--sensor-indices",
        type=str,
        default=None,
        help="CSV/range list (e.g. '4,12,20' or '0-63') or path to JSON list.",
    )

    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--ini", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _latest_episode(model_dir: str) -> int:
    actor_ckpts = [
        f
        for f in os.listdir(model_dir)
        if f.endswith("_actor.pt") and not f.endswith("_target_actor.pt")
    ]
    if not actor_ckpts:
        raise FileNotFoundError(f"No actor checkpoints found in {model_dir}")
    return max(int(f.split("_")[0]) for f in actor_ckpts)


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_dir = args.model_dir or f"./Model_{args.exp_name}"
    buffer_dir = args.buffer_dir or f"./Buffer_{args.exp_name}"

    x = np.loadtxt(args.x_file)
    if x.ndim != 1:
        raise ValueError(f"x-file must be 1D, got shape {x.shape}")

    u_target = np.loadtxt(args.target_file)
    if u_target.ndim != 1 or u_target.size != x.size:
        raise ValueError(
            f"target-file length must match x length ({x.size}), got shape {u_target.shape}"
        )

    ks = KS(L=args.domain_length, N=x.size, a_dim=args.a_dim)
    sensor_indices = _build_sensor_indices(args, x_size=x.size)
    s_dim = int(sensor_indices.size)

    print("State Dimensions :-", s_dim)
    print("Action Dimensions :-", args.a_dim)
    print("Action Max :-", args.a_max)
    print("Sensor Indices :-", sensor_indices.tolist())
    print("Sensor Positions :-", x[sensor_indices].tolist())
    print("Model Dir :-", model_dir)
    print("Buffer Dir :-", buffer_dir)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(buffer_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ram = buffer.MemoryBuffer(buffer_dir=buffer_dir)
    trainer = train.Trainer(
        s_dim,
        args.a_dim,
        args.a_max,
        ram,
        device,
        args.test,
        model_dir=model_dir,
        buffer_dir=buffer_dir,
    )

    init_states = None
    if args.test:
        init_states = np.loadtxt(args.test_init_file)
        if init_states.ndim == 1:
            init_states = init_states.reshape(1, -1)
        if init_states.shape[1] != x.size:
            raise ValueError(
                f"{args.test_init_file} row length ({init_states.shape[1]}) must match x size ({x.size})"
            )

    ini = args.ini
    if args.restart or args.test:
        if args.test and not os.path.exists(os.path.join(model_dir, f"{ini}_actor.pt")):
            ini = _latest_episode(model_dir)
            print("Test checkpoint not found for ini, using latest episode:", ini)
        trainer.load_models(ini, args.test)

    for ep in range(ini, args.max_episodes):
        if args.test:
            assert init_states is not None
            init_idx = np.random.randint(init_states.shape[0])
            new_observation = np.float32(init_states[init_idx].copy())
        else:
            new_observation = np.float32(np.loadtxt(args.train_init_file))

        reward = 0.0
        observation = new_observation
        for step in range(args.max_steps):
            state = np.float32(new_observation[sensor_indices])
            observation = new_observation
            action = trainer.get_action(state, Test=args.test)
            new_observation = ks.advance(observation, action)
            reward = -float(np.linalg.norm(new_observation - u_target))
            new_state = np.float32(new_observation[sensor_indices])

            if reward < args.max_total_reward and not args.test:
                reward = -100.0
                trainer.ram.add(state, action, reward, new_state, args.test)
                break

            trainer.ram.add(state, action, reward, new_state, args.test)
            trainer.optimize(args.test)

        trainer.update_pert(args.test)
        gc.collect()

        print(
            "EPISODE :-",
            ep,
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

        if ep % args.save_every == 0 and not args.test:
            trainer.save_models(ep)
            np.savetxt(os.path.join(buffer_dir, "state.dat"), observation)
            np.savetxt(os.path.join(buffer_dir, "action.dat"), ks.f0)
            np.savetxt(os.path.join(buffer_dir, "new_state.dat"), new_observation)

    print("Completed episodes")


if __name__ == "__main__":
    main()

