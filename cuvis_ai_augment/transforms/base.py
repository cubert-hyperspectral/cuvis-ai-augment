"""Base class and registry for augmentation transforms.

Transforms are plain Python objects (not cuvis-ai Nodes). They are registered via the
``@register("Name")`` decorator and looked up by :class:`AugmentationCompose` at construct
time. The discovery mechanism mirrors cuvis-ai's ``NodeRegistry``: explicit, declarative,
populated by import — no entry-point magic, no filesystem walking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor

# Module-level registry. Populated by @register decorators when transform modules import.
TRANSFORM_REGISTRY: dict[str, type[Transform]] = {}


def register(name: str):  # noqa: ANN201 — decorator factory
    """Class decorator that registers a :class:`Transform` subclass under ``name``.

    Raises
    ------
    ValueError
        If ``name`` is already in :data:`TRANSFORM_REGISTRY`.
    """

    def deco(cls: type[Transform]) -> type[Transform]:
        if name in TRANSFORM_REGISTRY:
            existing = TRANSFORM_REGISTRY[name].__name__
            raise ValueError(
                f"Transform name {name!r} already registered "
                f"(existing: {existing}, new: {cls.__name__})."
            )
        TRANSFORM_REGISTRY[name] = cls
        return cls

    return deco


def build_transform(spec: dict[str, Any]) -> Transform:
    """Instantiate a transform from a spec dict ``{"type": <name>, **kwargs}``.

    Looks ``type`` up in :data:`TRANSFORM_REGISTRY` and passes the remaining kwargs to
    the transform's ``__init__``. Raises a descriptive :class:`ValueError` listing all
    registered names if the requested type is unknown.
    """
    if not isinstance(spec, dict):
        raise TypeError(f"Transform spec must be a dict, got {type(spec).__name__}.")
    if "type" not in spec:
        raise ValueError(f"Transform spec missing required 'type' key. Got: {spec!r}")

    spec = dict(spec)  # copy so we don't mutate caller's dict
    name = spec.pop("type")

    if name not in TRANSFORM_REGISTRY:
        available = sorted(TRANSFORM_REGISTRY.keys())
        raise ValueError(
            f"Unknown transform {name!r}. Available transforms: {available}. "
            f"If this transform comes from an external package, list its module path "
            f"in the 'extra_transform_modules' hparam of AugmentationCompose."
        )

    cls = TRANSFORM_REGISTRY[name]
    return cls(**spec)


class Transform(ABC):
    """Abstract base for augmentation transforms.

    A Transform takes a cube ``(B, H, W, C)`` torch.float32 tensor and an optional mask
    ``(B, H, W)`` tensor, and returns a (possibly modified) ``(cube, mask)`` pair. Each
    Transform owns one hparam: ``prob`` — the probability of application per sample.

    Subclasses implement :meth:`__call__`. They must apply the *same* per-sample random
    decision to the cube and the mask so that paired alignment is preserved.

    Randomness is drawn from a shared :class:`torch.Generator` passed in by the caller
    (:class:`AugmentationCompose`). Subclasses must not create their own RNG state.

    Wavelength awareness
    --------------------
    ``__call__`` accepts an optional ``wavelengths`` argument (a ``list[float]`` of
    per-band centre wavelengths in nanometres, length must equal ``cube.shape[-1]``).
    Most transforms ignore this argument; wavelength-aware transforms (e.g. illuminant
    jitter, atmospheric-band warp — planned for v0.3.0+) consume it to make
    physically-grounded decisions. The argument was introduced in v0.2.0 as a
    backwards-compatible default so the contract can evolve without an API break.
    """

    def __init__(self, prob: float = 0.5, **kwargs: Any) -> None:
        if not 0.0 <= float(prob) <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob!r}")
        self.prob = float(prob)
        # Subclasses may accept additional kwargs; ignore unrecognised ones quietly
        # at the base level so callers get a clear error from the subclass __init__.
        if kwargs:
            extra = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"{type(self).__name__} got unexpected keyword arguments: {extra}")

    @abstractmethod
    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
        wavelengths: list[float] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Apply the transform.

        Parameters
        ----------
        cube : Tensor
            ``(B, H, W, C)`` torch.float32.
        mask : Tensor or None
            ``(B, H, W)`` torch.int32 / torch.bool, or ``None``.
        rng : torch.Generator
            Shared RNG. All randomness must come from this generator.
        wavelengths : list[float] or None, optional
            Per-band centre wavelengths in nanometres, length ``C``. ``None`` means
            wavelength metadata isn't available. Wavelength-agnostic transforms ignore
            this argument; wavelength-aware transforms (v0.3.0+) consume it.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Transformed ``(cube, mask)``.
        """
        ...

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _validate_shapes(cube: Tensor, mask: Tensor | None) -> None:
        """Raise if ``cube`` / ``mask`` don't match the documented contract."""
        if cube.ndim != 4:
            raise ValueError(f"cube must be 4-D (B, H, W, C); got shape {tuple(cube.shape)}.")
        if mask is not None:
            if mask.ndim != 3:
                raise ValueError(f"mask must be 3-D (B, H, W); got shape {tuple(mask.shape)}.")
            if mask.shape[:3] != cube.shape[:3]:
                raise ValueError(
                    f"cube/mask spatial shapes must match. "
                    f"cube={tuple(cube.shape[:3])}, mask={tuple(mask.shape[:3])}."
                )

    @staticmethod
    def _validate_wavelengths(cube: Tensor, wavelengths: list[float] | None) -> None:
        """Raise if ``wavelengths`` is given but inconsistent with the cube's C axis.

        Wavelength-aware transforms call this from their ``__call__`` before consuming
        ``wavelengths``. Wavelength-agnostic transforms ignore the argument entirely.
        """
        if wavelengths is None:
            return
        n_bands = cube.shape[-1]
        if len(wavelengths) != n_bands:
            raise ValueError(
                f"len(wavelengths)={len(wavelengths)} does not match cube C={n_bands}."
            )

    def _draw_apply_mask(
        self, batch_size: int, rng: torch.Generator, device: torch.device
    ) -> Tensor:
        """Draw a ``(B,)`` bool tensor marking which samples to apply the transform to.

        Each sample is independently sampled from Bernoulli(self.prob).
        """
        if self.prob == 0.0:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        if self.prob == 1.0:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        # Use CPU rng then move — torch.Generator is device-bound and the caller's rng is
        # typically CPU. Drawing on CPU keeps determinism portable.
        u = torch.rand(batch_size, generator=rng)
        return (u < self.prob).to(device=device)
