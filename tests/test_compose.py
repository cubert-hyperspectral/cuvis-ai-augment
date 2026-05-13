"""Tests for AugmentationCompose orchestration."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage

from cuvis_ai_augment.node.compose import AugmentationCompose
from cuvis_ai_augment.transforms import TRANSFORM_REGISTRY


def test_empty_transforms_is_identity(make_cube):
    cube, mask = make_cube()
    node = AugmentationCompose(transforms=[], seed=0)
    out = node.forward(cube=cube, mask=mask)
    assert torch.equal(out["cube"], cube)
    assert torch.equal(out["mask"], mask)


def test_compose_equals_sequential_application(make_cube):
    cube, mask = make_cube(height=8, width=8)
    specs = [
        {"type": "RandomHorizontalFlip", "prob": 1.0},
        {"type": "RandomVerticalFlip", "prob": 1.0},
    ]
    composed = AugmentationCompose(transforms=specs, seed=42)
    out = composed.forward(cube=cube, mask=mask)
    expected_cube = torch.flip(torch.flip(cube, dims=(2,)), dims=(1,))
    expected_mask = torch.flip(torch.flip(mask, dims=(2,)), dims=(1,))
    assert torch.equal(out["cube"], expected_cube)
    assert torch.equal(out["mask"], expected_mask)


def test_execution_stages_train_only():
    assert AugmentationCompose.execution_stages == {ExecutionStage.TRAIN}


def test_available_transforms_lists_v1():
    names = AugmentationCompose.available_transforms()
    assert set(names) >= {
        "Random90Rotate",
        "RandomHorizontalFlip",
        "RandomSpatialCrop",
        "RandomVerticalFlip",
    }
    # Sorted, no duplicates.
    assert names == sorted(set(names))


def test_unknown_transform_error_lists_available():
    with pytest.raises(ValueError) as exc:
        AugmentationCompose(transforms=[{"type": "BogusTransform"}])
    msg = str(exc.value)
    assert "Unknown transform 'BogusTransform'" in msg
    # Available names should be present in the error message.
    assert "RandomHorizontalFlip" in msg


def test_mask_optional(make_cube):
    cube, _ = make_cube()
    node = AugmentationCompose(transforms=[{"type": "RandomHorizontalFlip", "prob": 1.0}], seed=0)
    out = node.forward(cube=cube, mask=None)
    assert "mask" not in out
    assert torch.equal(out["cube"], torch.flip(cube, dims=(2,)))


def test_extra_transform_modules_loads_external(make_cube):
    """An external module path in extra_transform_modules should register its transforms
    before the Compose Node tries to resolve names."""
    cube, mask = make_cube()
    # tests.fixtures.fake_transform defines a `@register("IdentityTransform")` Transform.
    node = AugmentationCompose(
        transforms=[{"type": "IdentityTransform"}],
        extra_transform_modules=["tests.fixtures.fake_transform"],
        seed=0,
    )
    out = node.forward(cube=cube, mask=mask)
    assert torch.equal(out["cube"], cube)
    assert torch.equal(out["mask"], mask)
    # The fixture transform should now be in the global registry.
    assert "IdentityTransform" in TRANSFORM_REGISTRY


def test_seed_determinism(make_cube):
    cube, mask = make_cube(height=8, width=8)
    specs = [
        {"type": "RandomHorizontalFlip", "prob": 0.5},
        {"type": "RandomVerticalFlip", "prob": 0.5},
    ]
    n1 = AugmentationCompose(transforms=specs, seed=7)
    n2 = AugmentationCompose(transforms=specs, seed=7)
    o1 = n1.forward(cube=cube, mask=mask)
    o2 = n2.forward(cube=cube, mask=mask)
    assert torch.equal(o1["cube"], o2["cube"])
    assert torch.equal(o1["mask"], o2["mask"])
