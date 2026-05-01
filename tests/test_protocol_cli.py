import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.evaluate_sensor_count as evaluate_sensor_count
import scripts.train_run as train_run


class ProtocolCliTests(unittest.TestCase):
    def test_train_run_defaults_to_u3_and_shared_init_protocol(self) -> None:
        with patch.object(sys, "argv", ["train_run.py"]):
            args = train_run.parse_args()

        self.assertEqual(Path(args.target_file).name, "u3.dat")
        self.assertEqual(Path(args.init_file).name, "INIT.dat")
        self.assertEqual(Path(args.init_split_file).name, "controller_setup_split.json")
        self.assertEqual(args.split_seed, 123)
        self.assertEqual(args.train_split_size, 20)
        self.assertEqual(args.val_split_size, 20)
        self.assertEqual(args.test_split_size, 30)
        self.assertEqual(args.max_episodes, 1000)
        self.assertIsNone(args.reset_seed)

    def test_eval_defaults_to_fixed_test_split(self) -> None:
        with patch.object(sys, "argv", ["evaluate_sensor_count.py", "--models-spec", "models.json", "--dwell-time", "1.0"]):
            args = evaluate_sensor_count.parse_args()

        self.assertEqual(Path(args.target_file).name, "u3.dat")
        self.assertEqual(Path(args.init_file).name, "INIT.dat")
        self.assertEqual(Path(args.init_split_file).name, "controller_setup_split.json")
        self.assertEqual(args.split_role, "test")
        self.assertEqual(args.epsilon_mode, "target_relative")
        self.assertEqual(args.max_steps, 3000)


if __name__ == "__main__":
    unittest.main()
