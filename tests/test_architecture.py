"""Guard the ability to deploy localisation without the simulation stack."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class AlgorithmDependencyTests(unittest.TestCase):
    def test_algorithm_package_imports_without_simulator_or_visualisation_dependencies(self):
        script = textwrap.dedent("""
            import importlib
            import importlib.abc
            import pkgutil
            import sys

            blocked = {"simulator", "simulation", "environment", "network", "matplotlib", "shapely", "yaml"}

            class NoSimulatorImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split(".")[0] in blocked:
                        raise ImportError(f"Algorithm depends on simulation module: {fullname}")
                    return None

            sys.meta_path.insert(0, NoSimulatorImports())
            import crew_tracking
            for module in pkgutil.walk_packages(crew_tracking.__path__, prefix="crew_tracking."):
                importlib.import_module(module.name)
            assert not blocked.intersection(name.split(".")[0] for name in sys.modules)
        """)
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
