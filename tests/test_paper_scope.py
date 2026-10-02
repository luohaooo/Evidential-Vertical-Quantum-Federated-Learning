import unittest

from models import TeleportedRandomFixed, build_model
from paper_config import PAPER_TASKS, models_for_task


class PaperScopeTest(unittest.TestCase):
    def test_only_paper_tasks_are_exposed(self):
        self.assertEqual(
            tuple(PAPER_TASKS),
            (
                "mnist36",
                "mnist2356",
                "fashion19",
                "fashion1349",
                "breast",
                "credit",
            ),
        )

    def test_extra_teleported_baselines_are_limited_to_mnist36(self):
        self.assertIn("teleported-vqc", models_for_task("mnist36"))
        self.assertIn("teleported-random-fixed", models_for_task("mnist36"))
        for task in PAPER_TASKS:
            if task != "mnist36":
                self.assertNotIn("teleported-vqc", models_for_task(task))
                self.assertNotIn("teleported-random-fixed", models_for_task(task))

    def test_random_fixed_server_values_are_buffers_not_parameters(self):
        model = build_model("mnist36", "teleported-random-fixed", random_seed=7)
        self.assertIsInstance(model, TeleportedRandomFixed)
        parameter_names = dict(model.named_parameters())
        buffer_names = dict(model.named_buffers())
        self.assertNotIn("vqc.fixed_angles", parameter_names)
        self.assertIn("vqc.fixed_angles", buffer_names)
        self.assertIn("vqc.fixed_permutations", buffer_names)


if __name__ == "__main__":
    unittest.main()
