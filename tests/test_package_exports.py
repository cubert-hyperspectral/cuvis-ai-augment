"""Smoke: public top-level exports resolve."""

from __future__ import annotations


def test_top_level_exports() -> None:
    import cuvis_ai_augment

    assert cuvis_ai_augment.AugmentationCompose.__name__ == "AugmentationCompose"
    assert callable(cuvis_ai_augment.register)
    assert "Transform" in dir(cuvis_ai_augment)
    assert isinstance(cuvis_ai_augment.TRANSFORM_REGISTRY, dict)


def test_node_module_re_export() -> None:
    from cuvis_ai_augment.node import AugmentationCompose

    assert AugmentationCompose.__name__ == "AugmentationCompose"
