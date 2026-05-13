"""Transforms package. Importing this module populates :data:`TRANSFORM_REGISTRY`.

To add a new transforms module, import it here so its ``@register`` decorators run.
"""

from cuvis_ai_augment.transforms import spatial  # noqa: F401 — side-effect import
from cuvis_ai_augment.transforms.base import (
    TRANSFORM_REGISTRY,
    Transform,
    build_transform,
    register,
)

__all__ = ["TRANSFORM_REGISTRY", "Transform", "build_transform", "register"]
