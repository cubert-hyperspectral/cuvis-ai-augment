"""AugmentationCompose — the only Node this plugin exposes.

A single Node that applies a configurable list of stochastic Transforms to a cube (and
its paired mask) in order. Augmentations are sequenced via this node's ``transforms``
hparam rather than via separate pipeline edges, matching the
albumentations / torchvision.transforms.v2 / Kornia idiom.

``execution_stages={ExecutionStage.TRAIN}`` makes this node automatically a no-op at
val/test/inference — no extra wiring needed in pipeline YAMLs.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

# Import triggers @register decorators that populate TRANSFORM_REGISTRY.
from cuvis_ai_augment.transforms import (
    TRANSFORM_REGISTRY,
    build_transform,
)


class AugmentationCompose(Node):
    """Apply a sequence of stochastic augmentation Transforms to a cube + paired mask.

    Parameters
    ----------
    transforms : list[dict]
        Ordered list of transform specs. Each spec has a ``type`` key naming a
        registered Transform plus that transform's kwargs::

            transforms:
              - {type: RandomHorizontalFlip, prob: 0.5}
              - {type: RandomSpatialCrop,    size: [256, 256]}

        Unknown ``type`` raises :class:`ValueError` listing all registered names.

    seed : int or None
        Seed for the internal :class:`torch.Generator`. ``None`` means non-deterministic.

    extra_transform_modules : list[str]
        Optional list of importable Python module paths to import before resolving
        ``transforms``. Lets external packages contribute transforms via the same
        ``@register`` decorator without modifying this plugin.

    Notes
    -----
    The node accepts an optional ``mask`` port. When the mask is connected, each
    Transform applies the *same* per-sample random decision to both the cube and the
    mask so the pair stays aligned.
    """

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C] in float32",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Per-pixel mask [B, H, W] (int32 or bool)",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Augmented cube [B, H, W, C]",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Augmented mask [B, H, W]",
            optional=True,
        ),
    }

    # Augmentations only apply during training.
    execution_stages = {ExecutionStage.TRAIN}

    def __init__(
        self,
        transforms: list[dict[str, Any]] | None = None,
        seed: int | None = None,
        extra_transform_modules: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.transforms_spec: list[dict[str, Any]] = list(transforms or [])
        self.seed = seed
        self.extra_transform_modules: list[str] = list(extra_transform_modules or [])

        super().__init__(
            transforms=self.transforms_spec,
            seed=seed,
            extra_transform_modules=self.extra_transform_modules,
            **kwargs,
        )

        # Import any user-supplied transform modules so their @register decorators run
        # before we try to resolve names from TRANSFORM_REGISTRY.
        for module_path in self.extra_transform_modules:
            importlib.import_module(module_path)

        # Build the Transform instances once, fail fast on bad specs.
        self._transforms = [build_transform(spec) for spec in self.transforms_spec]

        # Seeded RNG, stored as a buffer-ish attribute. torch.Generator is not an
        # nn.Module so we keep a plain attribute and re-seed it lazily on first use
        # so device-moves don't need to migrate generator state (CPU rng works fine).
        self._rng = torch.Generator()
        if self.seed is not None:
            self._rng.manual_seed(int(self.seed))

    # -------------------------------------------------------------- introspection

    @classmethod
    def available_transforms(cls) -> list[str]:
        """Return the sorted list of currently registered transform names."""
        return sorted(TRANSFORM_REGISTRY.keys())

    # ----------------------------------------------------------------- forward

    def forward(
        self,
        cube: Tensor,
        mask: Tensor | None = None,
        **_: Any,
    ) -> dict[str, Tensor | None]:
        for transform in self._transforms:
            cube, mask = transform(cube, mask, self._rng)
        out: dict[str, Tensor | None] = {"cube": cube}
        if mask is not None:
            out["mask"] = mask
        return out
