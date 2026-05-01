import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_placement_campaign as run_placement_campaign


class RunPlacementCampaignTests(unittest.TestCase):
    def test_parse_args_defaults_to_three_rl_seeds(self) -> None:
        with patch.object(sys, "argv", ["run_placement_campaign.py"]):
            args = run_placement_campaign.parse_args()

        self.assertEqual(Path(args.target_file).name, "u3.dat")
        self.assertEqual(Path(args.init_file).name, "INIT.dat")
        self.assertEqual(Path(args.init_split_file).name, "controller_setup_split.json")
        self.assertEqual(args.train_seeds, "0,1,2")
        self.assertEqual(args.max_episodes, 1000)

    def test_parse_seed_csv_keeps_unique_order(self) -> None:
        self.assertEqual(run_placement_campaign.parse_seed_csv("0,2,2,1"), [0, 2, 1])

    def test_build_train_command_forwards_split_and_seed_protocol(self) -> None:
        args = run_placement_campaign.parse_args.__wrapped__ if hasattr(run_placement_campaign.parse_args, "__wrapped__") else None
        with patch.object(
            sys,
            "argv",
            [
                "run_placement_campaign.py",
                "--target-file",
                "data/u3.dat",
                "--init-file",
                "data/INIT.dat",
                "--init-split-file",
                "studies/controller_protocol/manifests/controller_setup_split.json",
                "--split-seed",
                "123",
                "--train-split-size",
                "20",
                "--val-split-size",
                "20",
                "--test-split-size",
                "30",
                "--reset-seed",
                "1000",
            ],
        ):
            parsed = run_placement_campaign.parse_args()

        cmd = run_placement_campaign.build_train_command(
            args=parsed,
            train_script=Path("scripts/train_run.py"),
            run_name="layout_seed0",
            run_root=Path("bundle/runs/layout/u3/seed_0"),
            model_dir=Path("bundle/runs/layout/u3/seed_0/model"),
            buffer_dir=Path("bundle/runs/layout/u3/seed_0/buffer"),
            sensor_indices_file=Path("bundle/runs/layout/u3/seed_0/sensor_indices.json"),
            state_dim=8,
            train_seed=0,
            reset_seed=1000,
            setup_name="layout",
        )

        command_text = " ".join(cmd)
        self.assertIn("--setup-name layout", command_text)
        self.assertIn("--init-file data/INIT.dat", command_text)
        self.assertIn("--init-split-file studies/controller_protocol/manifests/controller_setup_split.json", command_text)
        self.assertIn("--split-seed 123", command_text)
        self.assertIn("--train-split-size 20", command_text)
        self.assertIn("--val-split-size 20", command_text)
        self.assertIn("--test-split-size 30", command_text)
        self.assertIn("--train-seed 0", command_text)
        self.assertIn("--reset-seed 1000", command_text)
        self.assertNotIn("--train-init-file", command_text)
        self.assertNotIn("--val-init-file", command_text)


if __name__ == "__main__":
    unittest.main()
