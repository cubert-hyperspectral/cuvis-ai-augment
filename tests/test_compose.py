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


def test_execution_stages_always_routable_by_default():
    # ALWAYS-routable: the node stays in cuvis-ai-core's executable set at every
    # stage so downstream consumers always have a producer for our output port.
    # TRAIN-only behavior is enforced inside forward() via a stage check.
    node = AugmentationCompose(transforms=[], seed=0)
    assert node.execution_stages == {ExecutionStage.ALWAYS}
    assert node.should_execute(ExecutionStage.TRAIN) is True
    assert node.should_execute(ExecutionStage.VAL) is True
    assert node.should_execute(ExecutionStage.TEST) is True
    assert node.should_execute(ExecutionStage.INFERENCE) is True


def test_execution_stages_respect_explicit_override():
    # Callers can still pass execution_stages= explicitly; we honor it but warn,
    # because it re-introduces the filter-then-no-route issue and is not recommended.
    with pytest.warns(UserWarning, match="execution_stages"):
        node = AugmentationCompose(
            transforms=[],
            seed=0,
            execution_stages={ExecutionStage.TRAIN},
        )
    assert node.execution_stages == {ExecutionStage.TRAIN}
    assert node.should_execute(ExecutionStage.VAL) is False


def test_forward_is_identity_outside_train(make_cube):
    # At val/test/inference, forward must return cube/mask byte-identical to the
    # inputs even if non-trivial transforms are configured. This is the contract
    # the README advertises.
    from cuvis_ai_schemas.execution import Context

    cube, mask = make_cube(height=16, width=16)
    node = AugmentationCompose(
        transforms=[
            {"type": "RandomHorizontalFlip", "prob": 1.0},  # would always flip
            {"type": "RandomVerticalFlip", "prob": 1.0},
        ],
        seed=42,
    )
    for stage in (ExecutionStage.VAL, ExecutionStage.TEST, ExecutionStage.INFERENCE):
        ctx = Context(stage=stage)
        out = node.forward(cube=cube, mask=mask, context=ctx)
        assert torch.equal(out["cube"], cube), f"forward at {stage} not identity for cube"
        assert torch.equal(out["mask"], mask), f"forward at {stage} not identity for mask"


def test_forward_applies_transforms_at_train(make_cube):
    # At TRAIN the configured transforms must actually fire. Verifies the stage
    # check doesn't swallow training-time behavior.
    from cuvis_ai_schemas.execution import Context

    cube, mask = make_cube(height=16, width=16)
    node = AugmentationCompose(
        transforms=[{"type": "RandomHorizontalFlip", "prob": 1.0}],
        seed=0,
    )
    out = node.forward(cube=cube, mask=mask, context=Context(stage=ExecutionStage.TRAIN))
    assert torch.equal(out["cube"], torch.flip(cube, dims=(2,)))
    assert torch.equal(out["mask"], torch.flip(mask, dims=(2,)))


def test_forward_applies_transforms_when_no_context(make_cube):
    # Backwards-compatible: calling forward without a Context (e.g. unit tests that
    # invoke the node directly outside a pipeline) still applies transforms — the
    # "this is a transform; calling it transforms" intuition. Pipeline-driven calls
    # always pass context, so the stage gate only fires in production paths.
    cube, mask = make_cube(height=16, width=16)
    node = AugmentationCompose(
        transforms=[{"type": "RandomHorizontalFlip", "prob": 1.0}],
        seed=0,
    )
    out = node.forward(cube=cube, mask=mask)  # no context kwarg
    assert torch.equal(out["cube"], torch.flip(cube, dims=(2,)))
    assert torch.equal(out["mask"], torch.flip(mask, dims=(2,)))


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


def test_wavelengths_threaded_to_transforms(make_cube):
    """AugmentationCompose(wavelengths=...) must thread them to each transform's
    __call__ so wavelength-aware transforms (v0.3.0+) can consume the metadata."""
    from cuvis_ai_augment.transforms.base import Transform, register

    captured: list[list[float] | None] = []

    @register("_WavelengthCapturer")
    class _Capturer(Transform):
        def __call__(  # type: ignore[override]
            self,
            cube,
            mask,
            rng,
            wavelengths=None,
        ):
            captured.append(wavelengths)
            return cube, mask

    try:
        cube, mask = make_cube(channels=4)
        node = AugmentationCompose(
            transforms=[{"type": "_WavelengthCapturer"}],
            wavelengths=[500.0, 600.0, 700.0, 800.0],
            seed=0,
        )
        node.forward(cube=cube, mask=mask)
        assert captured == [[500.0, 600.0, 700.0, 800.0]]
    finally:
        TRANSFORM_REGISTRY.pop("_WavelengthCapturer", None)


def test_wavelengths_default_is_none(make_cube):
    """If user doesn't pass wavelengths, transforms see None — not an empty list."""
    from cuvis_ai_augment.transforms.base import Transform, register

    captured: list[list[float] | None] = []

    @register("_WavelengthCapturer2")
    class _Capturer2(Transform):
        def __call__(  # type: ignore[override]
            self,
            cube,
            mask,
            rng,
            wavelengths=None,
        ):
            captured.append(wavelengths)
            return cube, mask

    try:
        cube, mask = make_cube(channels=4)
        node = AugmentationCompose(transforms=[{"type": "_WavelengthCapturer2"}], seed=0)
        node.forward(cube=cube, mask=mask)
        assert captured == [None]
    finally:
        TRANSFORM_REGISTRY.pop("_WavelengthCapturer2", None)


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
