from __future__ import annotations

import unittest
from pathlib import Path


class ThirdPartyAttributionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def _text(self, relative: str) -> str:
        return (self.repo_root / relative).read_text(encoding="utf-8").lower()

    def test_third_party_notice_lists_upstreams_licenses_and_local_changes(self) -> None:
        text = self._text("THIRD_PARTY_NOTICES.md")

        expectations = {
            "deprl": ("depRL/", "https://github.com/martius-lab/depRL", "mit", "local msk-bench changes"),
            "dynsyn": ("msgym/", "https://github.com/Beanpow/DynSyn", "apache-2.0", "local msk-bench changes"),
            "musclemimic": (
                "third_party/musclemimic/",
                "https://github.com/amathislab/musclemimic",
                "apache-2.0",
                "local msk-bench changes",
            ),
            "tonic": ("depRL/deprl/vendor/tonic/", "https://github.com/fabiopardo/tonic", "mit", "vendored"),
            "ms-human-700": ("msk_bench/simhive/ms_human_700/", "ms-human-700", "license", "model assets"),
        }
        for component, required_phrases in expectations.items():
            self.assertIn(component, text)
            for phrase in required_phrases:
                self.assertIn(phrase.lower(), text, f"{component}: {phrase}")

    def test_modified_baseline_readmes_have_integration_attribution(self) -> None:
        for relative in (
            "depRL/README.md",
            "msgym/README.md",
            "deprl_middleware_22tasks/README.md",
        ):
            text = self._text(relative)
            self.assertIn("upstream", text, relative)
            self.assertIn("license", text, relative)
            self.assertIn("local msk-bench changes", text, relative)
            self.assertIn("do not remove", text, relative)

    def test_third_party_musclemimic_has_local_integration_readme(self) -> None:
        path = self.repo_root / "third_party" / "musclemimic" / "README.MSK-Bench.md"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8").lower()
        self.assertIn("https://github.com/amathislab/musclemimic", text)
        self.assertIn("apache-2.0", text)
        self.assertIn("local msk-bench changes", text)
        self.assertIn("do not remove", text)

    def test_local_patch_summary_exists(self) -> None:
        text = self._text("PATCHES.md")
        for phrase in (
            "depRL/",
            "msgym/",
            "third_party/musclemimic/",
            "local msk-bench changes",
            "upstream",
        ):
            self.assertIn(phrase.lower(), text)


if __name__ == "__main__":
    unittest.main()
