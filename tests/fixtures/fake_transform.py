"""External-package simulation: a Transform registered from outside cuvis_ai_augment.

Used by test_compose.test_extra_transform_modules_loads_external to verify the
``extra_transform_modules`` hparam path on :class:`AugmentationCompose`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from cuvis_ai_augment.transforms.base import Transform, register


@register("IdentityTransform")
class IdentityTransform(Transform):
    """No-op transform used as an external-registration smoke test."""

    def __init__(self, prob: float = 1.0, **kwargs: Any) -> None:
        super().__init__(prob=prob, **kwargs)

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
    ) -> tuple[Tensor, Tensor | None]:
        return cube, mask
