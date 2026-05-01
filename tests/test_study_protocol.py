import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ks_control.study_protocol import (  # type: ignore[attr-defined]
    build_study_split,
    compute_target_relative_epsilon,
    load_or_create_split_manifest,
    summarize_run_rows,
    summarize_setup_runs,
    validation_metrics_better,
)


class StudyProtocolTests(unittest.TestCase):
    def test_build_study_split_is_deterministic_and_non_overlapping(self) -> None:
        first = build_study_split(num_rows=100, split_seed=123, train_size=20, val_size=20, test_size=30)
        second = build_study_split(num_rows=100, split_seed=123, train_size=20, val_size=20, test_size=30)

        self.assertEqual(first, second)
        self.assertEqual(len(first["train_rows"]), 20)
        self.assertEqual(len(first["val_rows"]), 20)
        self.assertEqual(len(first["test_rows"]), 30)

        all_rows = first["train_rows"] + first["val_rows"] + first["test_rows"]
        self.assertEqual(len(all_rows), len(set(all_rows)))

    def test_load_or_create_split_manifest_persists_shared_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "init_split.json"
            payload = load_or_create_split_manifest(
                path=path,
                init_file="data/INIT.dat",
                num_rows=90,
                split_seed=7,
                train_size=20,
                val_size=20,
                test_size=30,
            )
            reloaded = load_or_create_split_manifest(
                path=path,
                init_file="data/INIT.dat",
                num_rows=90,
                split_seed=999,
                train_size=20,
                val_size=20,
                test_size=30,
            )

        self.assertEqual(payload, reloaded)
        self.assertEqual(payload["split_seed"], 7)
        self.assertEqual(payload["train_size"], 20)
        self.assertEqual(payload["val_size"], 20)
        self.assertEqual(payload["test_size"], 30)

    def test_load_or_create_split_manifest_accepts_windows_and_posix_init_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "init_split.json"
            payload = load_or_create_split_manifest(
                path=path,
                init_file="data\\INIT.dat",
                num_rows=90,
                split_seed=7,
                train_size=20,
                val_size=20,
                test_size=30,
            )
            reloaded = load_or_create_split_manifest(
                path=path,
                init_file="data/INIT.dat",
                num_rows=90,
                split_seed=999,
                train_size=20,
                val_size=20,
                test_size=30,
            )

        self.assertEqual(payload["init_file"], "INIT.dat")
        self.assertEqual(reloaded["init_file"], "INIT.dat")

    def test_validation_metrics_compare_success_then_error_then_effort(self) -> None:
        incumbent = {
            "success_rate": 0.80,
            "mean_final_error": 0.20,
            "mean_control_effort": 0.40,
        }
        better_success = {
            "success_rate": 0.85,
            "mean_final_error": 1.50,
            "mean_control_effort": 9.00,
        }
        better_error = {
            "success_rate": 0.80,
            "mean_final_error": 0.19,
            "mean_control_effort": 9.00,
        }
        better_effort = {
            "success_rate": 0.80,
            "mean_final_error": 0.20,
            "mean_control_effort": 0.39,
        }

        self.assertTrue(validation_metrics_better(better_success, incumbent))
        self.assertTrue(validation_metrics_better(better_error, incumbent))
        self.assertTrue(validation_metrics_better(better_effort, incumbent))
        self.assertFalse(validation_metrics_better(incumbent, incumbent))

    def test_compute_target_relative_epsilon_scales_target_norm(self) -> None:
        u_target = [3.0, 4.0]
        epsilon = compute_target_relative_epsilon(u_target=u_target, epsilon_beta=0.10)

        self.assertAlmostEqual(epsilon, 0.5)

    def test_run_and_setup_summaries_follow_two_stage_protocol(self) -> None:
        run_summary = summarize_run_rows(
            rows=[
                {
                    "success": True,
                    "final_error": 0.1,
                    "integrated_error": 1.0,
                    "control_effort": 0.4,
                    "time_to_stabilize": 4.0,
                },
                {
                    "success": False,
                    "final_error": 0.3,
                    "integrated_error": 1.4,
                    "control_effort": 0.6,
                    "time_to_stabilize": None,
                },
            ]
        )
        setup_summary = summarize_setup_runs(
            run_summaries=[
                {"success_rate": 0.5, "mean_final_error": 0.2, "mean_integrated_error": 1.2, "mean_control_effort": 0.5},
                {"success_rate": 0.75, "mean_final_error": 0.1, "mean_integrated_error": 0.9, "mean_control_effort": 0.3},
                {"success_rate": 1.0, "mean_final_error": 0.05, "mean_integrated_error": 0.8, "mean_control_effort": 0.2},
            ]
        )

        self.assertAlmostEqual(run_summary["success_rate"], 0.5)
        self.assertAlmostEqual(run_summary["mean_final_error"], 0.2)
        self.assertAlmostEqual(run_summary["mean_integrated_error"], 1.2)
        self.assertAlmostEqual(run_summary["mean_control_effort"], 0.5)
        self.assertAlmostEqual(run_summary["mean_tts_success_only"], 4.0)

        self.assertAlmostEqual(setup_summary["success_rate"]["mean"], 0.75)
        self.assertAlmostEqual(setup_summary["mean_final_error"]["median"], 0.1)
        self.assertAlmostEqual(setup_summary["mean_control_effort"]["mean"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
