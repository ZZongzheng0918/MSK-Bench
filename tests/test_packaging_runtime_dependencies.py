from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


class PackagingRuntimeDependenciesTest(unittest.TestCase):
    def test_default_install_includes_environment_runtime_dependencies(self) -> None:
        pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
        dependencies = "\n".join(pyproject["project"]["dependencies"]).lower()

        for package in ("gymnasium", "mujoco", "numpy", "scipy"):
            self.assertIn(package, dependencies)


if __name__ == "__main__":
    unittest.main()
