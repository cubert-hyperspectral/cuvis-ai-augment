"""Tests for the Transform base class, registry, and build_transform.

These cover the error branches that the V1 transforms can't reach on their own —
specifically: duplicate registration, malformed specs, out-of-range prob, and the
shape-validation contract.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from cuvis_ai_augment.transforms.base import (
    TRANSFORM_REGISTRY,
    Transform,
    build_transform,
    register,
)

# ---------------------------------------------------------------- registry


def test_register_duplicate_name_raises():
    """Registering a second class under the same name must fail loudly so users
    can't silently shadow a built-in transform."""
    sentinel = "_DuplicateRegisterSentinel"
    try:

        @register(sentinel)
        class _A(Transform):
            def __call__(self, cube, mask, rng):  # type: ignore[override]
                return cube, mask

        with pytest.raises(ValueError, match="already registered"):

            @register(sentinel)
            class _B(Transform):
                def __call__(self, cube, mask, rng):  # type: ignore[override]
                    return cube, mask
    finally:
        TRANSFORM_REGISTRY.pop(sentinel, None)


# ---------------------------------------------------------------- build_transform


def test_build_transform_rejects_non_dict_spec():
    with pytest.raises(TypeError, match="must be a dict"):
        build_transform("RandomHorizontalFlip")  # type: ignore[arg-type]


def test_build_transform_rejects_missing_type_key():
    with pytest.raises(ValueError, match="missing required 'type' key"):
        build_transform({"prob": 0.5})


def test_build_transform_unknown_type_lists_available():
    """The error message must include the registered transform names so the user
    can fix the typo without grepping the source."""
    with pytest.raises(ValueError) as exc:
        build_transform({"type": "NoSuchTransform"})
    msg = str(exc.value)
    assert "Unknown transform 'NoSuchTransform'" in msg
    # At minimum the four V1 names should be in the suggestion list.
    for name in (
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "Random90Rotate",
        "RandomSpatialCrop",
    ):
        assert name in msg


def test_build_transform_does_not_mutate_caller_spec():
    spec = {"type": "RandomHorizontalFlip", "prob": 1.0}
    snapshot = dict(spec)
    build_transform(spec)
    assert spec == snapshot, "build_transform must not mutate the caller's spec dict."


# ---------------------------------------------------------------- Transform.__init__


def test_prob_out_of_range_raises():
    with pytest.raises(ValueError, match=r"prob must be in \[0, 1\]"):
        build_transform({"type": "RandomHorizontalFlip", "prob": 1.5})
    with pytest.raises(ValueError, match=r"prob must be in \[0, 1\]"):
        build_transform({"type": "RandomHorizontalFlip", "prob": -0.1})


def test_unexpected_kwargs_raise():
    """A typo in a kwarg should fail at construction, not be silently ignored —
    otherwise users debug a transform that 'does nothing different'."""
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        build_transform({"type": "RandomHorizontalFlip", "prob": 0.5, "probabilty": 0.5})


# ---------------------------------------------------------------- shape validation


class _PassThrough(Transform):
    """Minimal Transform that just exposes _validate_shapes for testing."""

    def __call__(  # type: ignore[override]
        self,
        cube: torch.Tensor,
        mask: torch.Tensor | None,
        rng: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_shapes(cube, mask)
        return cube, mask


def _rng() -> torch.Generator:
    return torch.Generator().manual_seed(0)


def test_validate_shapes_rejects_3d_cube():
    cube = torch.zeros((4, 4, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="cube must be 4-D"):
        _PassThrough()(cube, None, _rng())


def test_validate_shapes_rejects_2d_mask():
    cube = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    bad_mask = torch.zeros((4, 4), dtype=torch.int32)
    with pytest.raises(ValueError, match="mask must be 3-D"):
        _PassThrough()(cube, bad_mask, _rng())


def test_validate_shapes_rejects_mismatched_spatial_dims():
    cube = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    bad_mask = torch.zeros((1, 5, 4), dtype=torch.int32)
    with pytest.raises(ValueError, match="cube/mask spatial shapes must match"):
        _PassThrough()(cube, bad_mask, _rng())


# ---------------------------------------------------------------- _draw_apply_mask


def test_draw_apply_mask_extremes_avoid_rng_draw():
    """prob=0 and prob=1 must return constant tensors without drawing from rng —
    that's a correctness property (deterministic identity / always-apply) and a
    micro-optimisation; the test pins both."""
    rng = _rng()

    t0: Any = build_transform({"type": "RandomHorizontalFlip", "prob": 0.0})
    apply0 = t0._draw_apply_mask(8, rng, torch.device("cpu"))
    assert apply0.dtype == torch.bool
    assert apply0.shape == (8,)
    assert not apply0.any()

    t1: Any = build_transform({"type": "RandomHorizontalFlip", "prob": 1.0})
    apply1 = t1._draw_apply_mask(8, rng, torch.device("cpu"))
    assert apply1.all()
