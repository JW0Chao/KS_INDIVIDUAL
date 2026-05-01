import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ks_control.evaluation_helpers import pick_checkpoint_episode


class EvaluationHelpersTests(unittest.TestCase):
    def test_pick_checkpoint_episode_ignores_best_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "0_actor.pt").write_text("", encoding="utf-8")
            (models_dir / "1_actor.pt").write_text("", encoding="utf-8")
            (models_dir / "best_actor.pt").write_text("", encoding="utf-8")

            episode = pick_checkpoint_episode(models_dir=models_dir, requested_episode=None, need_critic=False)

        self.assertEqual(episode, 1)


if __name__ == "__main__":
    unittest.main()
