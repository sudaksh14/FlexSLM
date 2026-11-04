"""Smoke test: every relocated module (training/, scripts/, eval/, data/) still imports."""
import importlib
import unittest

MODULES = [
    "training", "training.training", "training.training_memopt",
    "config.experiments",
    "scripts.run_experiment", "eval.eval_wikitext103",
    "data.prepare_fineweb",
]
OPTIONAL = ["eval.eval_lm_harness", "scripts.plot", "scripts.bench_memopt"]  # extra deps (lm_eval, mixbox, ...)


class TestImports(unittest.TestCase):
    def test_modules_import(self):
        for m in MODULES:
            with self.subTest(module=m):
                importlib.import_module(m)

    def test_optional_modules_import(self):
        for m in OPTIONAL:
            with self.subTest(module=m):
                try:
                    importlib.import_module(m)
                except ModuleNotFoundError as e:
                    self.skipTest(f"{m}: missing dependency {e.name}")

    def test_legacy_names_resolve(self):
        import training
        import sys
        self.assertIs(sys.modules["training_memopt"], training.training_memopt)
        self.assertTrue(callable(training._print_experiment_banner))


if __name__ == "__main__":
    unittest.main()
