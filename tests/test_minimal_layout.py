from __future__ import annotations

import unittest
from pathlib import Path


class MinimalLayoutTest(unittest.TestCase):
    def test_repository_uses_normalized_minimal_layout(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        self.assertTrue((repo_root / "msk_bench").is_dir())
        self.assertFalse((repo_root / ("paper" + "_reproduction")).exists())
        self.assertFalse((repo_root / "configs" / "paper").exists())
        self.assertFalse((repo_root / "msk_bench" / "reproduce.py").exists())
        self.assertFalse((repo_root / "MSK-Bench").exists())
        self.assertFalse((repo_root / "scripts").exists())
        self.assertFalse((repo_root / "draw_emg_12_muscles.py").exists())

    def test_open_source_metadata_files_exist(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        for relative in (
            "README.md",
            "pyproject.toml",
            ".gitignore",
            "CONTRIBUTING.md",
            "CITATION.cff",
            "THIRD_PARTY_NOTICES.md",
            "PATCHES.md",
            ".github/workflows/ci.yml",
        ):
            self.assertTrue((repo_root / relative).is_file(), relative)

    def test_baseline_directories_include_readmes(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        for relative in (
            "ppo/README.md",
            "sac/README.md",
            "depRL/README.md",
            "msgym/README.md",
            "deprl_middleware_22tasks/README.md",
            "benchmark_eval/README.md",
        ):
            self.assertTrue((repo_root / relative).is_file(), relative)

    def test_readme_omits_publication_artifact_docs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        text = (repo_root / "README.md").read_text(encoding="utf-8").lower()

        for forbidden in (
            "paper" + " reproduction",
            "reproduce" + "_figure",
            "table" + " ii",
            "data/" + "paper",
            "checkpoints/" + "figure",
        ):
            self.assertNotIn(forbidden, text)

    def test_no_python_cache_directories_are_kept(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        cache_dirs = [path for path in repo_root.rglob("__pycache__") if ".git" not in path.parts]
        self.assertEqual(cache_dirs, [])


if __name__ == "__main__":
    unittest.main()
