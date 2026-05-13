"""Plugin-manifest loading smoke test (mirrors dinomaly)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_examples_plugins_manifest_loads_and_registers_nodes() -> None:
    """NodeRegistry must load the local manifest and resolve AugmentationCompose."""
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    manifest = Path(__file__).resolve().parents[1] / "examples" / "plugins.yaml"
    registry = NodeRegistry()
    registry.load_plugins(str(manifest))

    cls = registry.get("AugmentationCompose")
    assert cls.__name__ == "AugmentationCompose"
