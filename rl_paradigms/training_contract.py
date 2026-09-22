"""Shared artifact contract for repository training entry points."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class TrainingArtifact:
    algorithm: str
    env_id: str
    seed: int
    timesteps: int
    model_path: Path
    normalization_path: Path | None = None

    def write(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["model_path"] = str(Path(self.model_path).resolve())
        if self.normalization_path is not None:
            payload["normalization_path"] = str(Path(self.normalization_path).resolve())
        path = output_dir / "training_manifest.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path
