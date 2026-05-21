"""AugmentationCompose — the only Node this plugin exposes.

A single Node that applies a configurable list of stochastic Transforms to a cube (and
its paired mask) in order. Augmentations are sequenced via this node's ``transforms``
hparam rather than via separate pipeline edges, matching the
albumentations / torchvision.transforms.v2 / Kornia idiom.

The node is registered as ``execution_stages={ALWAYS}`` so cuvis-ai-core keeps it in
the executable graph at every stage. TRAIN-only behavior is enforced *internally* in
``forward`` via a Context stage check: at val/test/inference it short-circuits to
identity (cube and mask passed through unchanged). This delivers the documented
"no-op at val/test/inference" semantics while preserving downstream port routing.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
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

    wavelengths : list[float] or None
        Optional per-band centre wavelengths in nanometres, length must equal ``C`` of
        the cube. Threaded through to every transform's ``__call__``. Wavelength-aware
        transforms (v0.3.0+) consume it; the v0.2.0 transforms ignore it. Set this
        once per pipeline — it's assumed constant across batches.

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

    # TRAIN-only behavior is enforced inside ``forward`` (Context stage check), NOT
    # via cuvis-ai-core's ``execution_stages`` filter, for two reasons:
    #
    # 1. ``Node.__init__`` writes ``self.execution_stages`` from a kwarg (defaulting
    #    to ``{ALWAYS}``) on every instance, *unconditionally* — a class-level
    #    declaration here would be silently shadowed at construction time and never
    #    reach ``should_execute``. Empirically verified (see issue #8).
    #
    # 2. Even if the class attribute were honored, cuvis-ai-core's pipeline reacts
    #    to a filtered node by removing it from the executable set entirely (no
    #    passthrough routing). Downstream consumers that subscribed to our output
    #    port would then crash with ``missing required input``. Keeping the node
    #    ALWAYS-routable and short-circuiting to identity in ``forward`` is the
    #    only way to deliver the "no-op at val/test/inference" behavior without a
    #    cuvis-ai-core change.
    def __init__(
        self,
        transforms: list[dict[str, Any]] | None = None,
        seed: int | None = None,
        extra_transform_modules: list[str] | None = None,
        wavelengths: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        self.transforms_spec: list[dict[str, Any]] = list(transforms or [])
        self.seed = seed
        self.extra_transform_modules: list[str] = list(extra_transform_modules or [])
        self.wavelengths: list[float] | None = (
            [float(w) for w in wavelengths] if wavelengths is not None else None
        )

        # ALWAYS-routable: node stays in the executable graph at all stages; stage
        # gating happens inside forward(). Callers can override by passing
        # execution_stages= explicitly, but that re-introduces the routing issue
        # described above (use at your own risk).
        kwargs.setdefault("execution_stages", {ExecutionStage.ALWAYS})

        super().__init__(
            transforms=self.transforms_spec,
            seed=seed,
            extra_transform_modules=self.extra_transform_modules,
            wavelengths=self.wavelengths,
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
        context: Context | None = None,
        **_: Any,
    ) -> dict[str, Tensor | None]:
        # Stage gate: identity passthrough at val/test/inference. When called outside
        # a cuvis-ai-core pipeline (context is None), default to applying transforms —
        # matches the "this is a transform; calling it transforms" intuition that
        # standalone unit tests rely on.
        stage = getattr(context, "stage", None) if context is not None else None
        if stage is not None and stage != ExecutionStage.TRAIN:
            out_identity: dict[str, Tensor | None] = {"cube": cube}
            if mask is not None:
                out_identity["mask"] = mask
            return out_identity

        for transform in self._transforms:
            cube, mask = transform(cube, mask, self._rng, wavelengths=self.wavelengths)
        out: dict[str, Tensor | None] = {"cube": cube}
        if mask is not None:
            out["mask"] = mask
        return out
