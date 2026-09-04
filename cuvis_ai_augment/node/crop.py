"""Crop — a deterministic, fixed-rectangle spatial crop Node.

The stochastic counterparts in this plugin (``RandomSpatialCrop``,
``RandomForegroundBiasedCrop``) live inside :class:`AugmentationCompose` and only
run at TRAIN. ``Crop`` is different: it is a *deterministic* preprocessing op that
selects a fixed sub-rectangle of every frame, so it must run identically at every
stage (train, val, test, inference). It is therefore a standalone Node on the class
default ``EXECUTION_STAGES = {ALWAYS}`` with **no** internal stage gate — cropping a
val/test cube is exactly the intended behavior, not something to short-circuit.

Bounds select ``data[:, top:bottom, left:right, :]``; ``None`` means open-ended,
so the defaults (``top=0, bottom=None, left=0, right=None``) are an identity
passthrough. Unlike raw Python slicing, the bounds are validated against the
actual frame at ``forward``: a rectangle that exceeds (or falls entirely outside)
the data raises ``ValueError`` instead of being silently clipped to a smaller —
or empty — result. When a ``mask`` port is connected its spatial shape must match
the cube's and it is cropped with the *same* rectangle so the pair stays aligned.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_augment.transforms.base import validate_cube_mask_shapes


class Crop(Node):
    """Deterministic fixed-rectangle crop of a ``[B, H, W, C]`` cube (and paired mask).

    Parameters
    ----------
    top, bottom : int or None
        Row bounds (``H`` axis). ``top`` defaults to 0, ``bottom`` to ``None``
        (frame height). Kept rows are ``[top:bottom]``. ``forward`` raises
        ``ValueError`` when ``bottom`` exceeds the actual frame height or the
        resolved rectangle is empty — bounds are never silently clipped.
    left, right : int or None
        Column bounds (``W`` axis). ``left`` defaults to 0, ``right`` to ``None``
        (frame width). Kept columns are ``[left:right]``, validated the same way.

    Notes
    -----
    The cropped tensors are returned ``.contiguous()`` because downstream nodes
    (normalizers, reshapes) frequently call ``.view()``, which requires a compact
    memory layout that a slice view does not guarantee.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.PREPROCESSING, NodeTag.HYPERSPECTRAL, NodeTag.TORCH})

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Cube [B, H, W, C] in float32",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Optional per-pixel mask [B, H, W] (cropped identically)",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "cropped": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Cropped cube [B, H', W', C]",
        ),
        "mask_cropped": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Cropped mask [B, H', W'] (only when a mask is connected)",
            optional=True,
        ),
    }

    def __init__(
        self,
        top: int = 0,
        bottom: int | None = None,
        left: int = 0,
        right: int | None = None,
        **kwargs: Any,
    ) -> None:
        top = int(top)
        left = int(left)
        bottom = None if bottom is None else int(bottom)
        right = None if right is None else int(right)
        if top < 0 or left < 0:
            raise ValueError(f"Crop: top/left must be >= 0, got top={top}, left={left}.")
        if bottom is not None and bottom <= top:
            raise ValueError(f"Crop: bottom ({bottom}) must be greater than top ({top}).")
        if right is not None and right <= left:
            raise ValueError(f"Crop: right ({right}) must be greater than left ({left}).")
        self.top, self.bottom, self.left, self.right = top, bottom, left, right
        super().__init__(top=top, bottom=bottom, left=left, right=right, **kwargs)

    def forward(
        self,
        data: Tensor,
        mask: Tensor | None = None,
        context: Context | None = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        # Same cube/mask contract checks every Transform in this plugin performs
        # (4-D cube, 3-D mask, matching spatial shapes) — shared helper, same messages.
        validate_cube_mask_shapes(data, mask)

        # Python slicing silently clips out-of-range indices, which would mask a
        # misconfigured rectangle (or return an empty tensor whose failure only
        # surfaces downstream). Validate the resolved bounds against the actual
        # frame and refuse, mirroring RandomSpatialCrop's size-vs-frame check.
        _, H, W, _ = data.shape
        bottom = self.bottom if self.bottom is not None else H
        right = self.right if self.right is not None else W
        if bottom > H or right > W:
            raise ValueError(
                f"Crop bounds exceed the data: bottom={bottom}, right={right} vs "
                f"frame (H={H}, W={W}). Bounds are not silently clipped."
            )
        if self.top >= bottom or self.left >= right:
            raise ValueError(
                f"Crop rectangle is empty for this frame: rows [{self.top}:{bottom}], "
                f"cols [{self.left}:{right}] on (H={H}, W={W})."
            )

        out: dict[str, Tensor] = {
            "cropped": data[:, self.top : bottom, self.left : right, :].contiguous()
        }
        if mask is not None:
            out["mask_cropped"] = mask[:, self.top : bottom, self.left : right].contiguous()
        return out
