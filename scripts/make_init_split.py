from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
DEFAULT_OUT = ROOT / "studies" / "controller_protocol" / "manifests" / "controller_setup_split.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ks_control.study_protocol import load_or_create_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or validate the shared INIT.dat study split.")
    parser.add_argument("--init-file", type=str, default=str(DATA_DIR / "INIT.dat"))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--val-size", type=int, default=20)
    parser.add_argument("--test-size", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_file = Path(args.init_file)
    if init_file.name != "INIT.dat":
        raise ValueError(f"init-file must point to INIT.dat, got '{init_file.name}'.")

    init_states = np.loadtxt(init_file)
    if init_states.ndim == 1:
        init_states = init_states.reshape(1, -1)

    payload = load_or_create_split_manifest(
        path=Path(args.out),
        init_file=str(init_file),
        num_rows=int(init_states.shape[0]),
        split_seed=int(args.split_seed),
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
    )
    print(f"Split manifest ready: {args.out}")
    print(
        f"train={payload['train_size']} val={payload['val_size']} test={payload['test_size']} "
        f"split_seed={payload['split_seed']}"
    )


if __name__ == "__main__":
    main()
