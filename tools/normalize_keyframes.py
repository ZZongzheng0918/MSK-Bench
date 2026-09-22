"""Normalize MJCF keyframe qpos arrays against the compiled model size."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


KEY_QPOS = re.compile(r'(<key\b[^>]*?\bqpos=")([^"]*)(")')


def normalized_qpos(
    raw: str,
    qpos0: np.ndarray,
    *,
    allow_truncate: bool,
) -> list[float]:
    """Return exactly ``nq`` values, padding from the model's qpos0."""
    values = [float(value) for value in raw.split()]
    if len(values) > qpos0.size and not allow_truncate:
        raise ValueError(
            f"keyframe has {len(values)} qpos values; model nq={qpos0.size}"
        )
    return (values + qpos0[len(values) :].tolist())[: qpos0.size]


def _model_without_key_qpos(path: Path, text: str) -> mujoco.MjModel:
    """Compile a sibling copy so relative MJCF includes keep working."""

    def remove_qpos(match: re.Match[str]) -> str:
        tag = match.group(0)
        return re.sub(r'\s+qpos="[^"]*"', "", tag)

    stripped = re.sub(r'<key\b[^>]*>', remove_qpos, text)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-no-key-qpos-",
        suffix=path.suffix,
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(stripped)
        return mujoco.MjModel.from_xml_path(str(temporary))
    finally:
        temporary.unlink(missing_ok=True)


def normalize_file(
    path: Path,
    *,
    allow_truncate: bool,
    check: bool,
) -> bool:
    """Normalize one file and return whether it required a change."""
    text = path.read_text(encoding="utf-8")
    root = ET.fromstring(text)
    xml_keys = [key for key in root.findall(".//key") if "qpos" in key.attrib]
    matches = list(KEY_QPOS.finditer(text))
    if len(matches) != len(xml_keys):
        raise ValueError(
            f"{path}: parsed {len(xml_keys)} qpos keys but matched {len(matches)}"
        )

    model = _model_without_key_qpos(path, text)
    qpos0 = np.asarray(model.qpos0, dtype=float)
    old_lengths = [len(match.group(2).split()) for match in matches]
    changed = any(length != model.nq for length in old_lengths)
    print(f"{path}: nq={model.nq}, key qpos lengths={old_lengths}")
    if not changed or check:
        return changed

    def replace_qpos(match: re.Match[str]) -> str:
        values = normalized_qpos(
            match.group(2),
            qpos0,
            allow_truncate=allow_truncate,
        )
        rendered = " ".join(f"{value:.15g}" for value in values)
        return f"{match.group(1)}{rendered}{match.group(3)}"

    normalized = KEY_QPOS.sub(replace_qpos, text)
    temporary = path.with_name(f".{path.name}.normalized")
    temporary.write_text(normalized, encoding="utf-8")
    os.replace(temporary, path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--allow-truncate", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    changed = False
    for path in args.paths:
        try:
            changed |= normalize_file(
                path,
                allow_truncate=args.allow_truncate,
                check=args.check,
            )
        except (OSError, ValueError, ET.ParseError) as error:
            parser.error(str(error))
    return int(args.check and changed)


if __name__ == "__main__":
    raise SystemExit(main())
