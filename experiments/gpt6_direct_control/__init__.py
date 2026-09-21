"""Paused direct-muscle-control experiment infrastructure.

This package contains no LLM client and never calls an external language-model API.
The Phase 2 controller is the active Codex model, operating through persisted JSON.
"""

EXPERIMENT_VERSION = "1.0"

__all__ = ["EXPERIMENT_VERSION"]
