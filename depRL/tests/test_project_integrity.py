import ast
import re
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_msk_bench_package():
    candidates = [
        ROOT / "msk_bench",
        ROOT.parent / "msk_bench",
        ROOT.parent / "MSK-Bench" / "msk_bench",
    ]
    for candidate in candidates:
        if (candidate / "__init__.py").exists():
            return candidate
    return ROOT / "msk_bench"


PACKAGE = resolve_msk_bench_package()
BENCHMARK_INIT = PACKAGE / "envs" / "msk" / "benchmark" / "__init__.py"
BENCHMARK_IMPL = PACKAGE / "envs" / "msk" / "benchmark" / "msk_bench_v0.py"
BODY_DIR = PACKAGE / "simhive" / "msk_sim" / "body"


class ProjectIntegrityTests(unittest.TestCase):
    def test_smoke_test_dependencies_are_declared(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        dep_block = re.search(r"dependencies\s*=\s*\[(.*?)\]", pyproject, re.S)
        self.assertIsNotNone(dep_block)
        deps = {
            re.split(r"[<>=!~]", dep)[0].lower().replace("_", "-")
            for dep in re.findall(r'"([^"\n]+)"', dep_block.group(1))
        }
        self.assertIn("imageio", deps)

    def test_generated_smoke_gifs_are_ignored(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertRegex(gitignore, r"(?m)^\*\.gif$")

    def test_registered_msk_bench_tasks_have_classes_and_models(self):
        init_text = BENCHMARK_INIT.read_text(encoding="utf-8")
        impl_tree = ast.parse(BENCHMARK_IMPL.read_text(encoding="utf-8"))
        classes = {node.name for node in ast.walk(impl_tree) if isinstance(node, ast.ClassDef)}
        registrations = re.findall(
            r'register_msk_bench_task\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"',
            init_text,
        )
        self.assertEqual(len(registrations), 22)
        for task_name, class_name, model_file in registrations:
            with self.subTest(task=task_name):
                self.assertIn(class_name, classes)
                self.assertTrue((BODY_DIR / model_file).exists(), model_file)

    def test_registered_model_include_files_exist(self):
        init_text = BENCHMARK_INIT.read_text(encoding="utf-8")
        model_files = sorted(set(re.findall(r'"([^"]+\.xml)"', init_text)))
        for model_file in model_files:
            with self.subTest(model=model_file):
                self._assert_includes_exist(BODY_DIR / model_file)

    def _assert_includes_exist(self, model_path):
        main_dir = model_path.parent
        seen = set()

        def walk(path):
            path = path.resolve()
            if path in seen:
                return
            seen.add(path)
            root = ET.parse(path).getroot()
            for elem in root.iter():
                if elem.tag.split("}")[-1] != "include":
                    continue
                include_file = elem.attrib["file"]
                include_path = (main_dir / include_file).resolve()
                self.assertTrue(include_path.exists(), f"{path} includes missing {include_file}")
                walk(include_path)

        walk(model_path)


if __name__ == "__main__":
    unittest.main()
