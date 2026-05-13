"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
import torch


def pytest_configure(config: object) -> None:
    config.addinivalue_line("markers", "slow: long-running or heavy")
    config.addinivalue_line("markers", "integration: full stack / plugin manifest")


@pytest.fixture
def make_cube():
    """Factory: build a deterministic (B, H, W, C) float32 cube and (B, H, W) mask."""

    def _make(
        batch_size: int = 2,
        height: int = 16,
        width: int = 16,
        channels: int = 8,
        with_mask: bool = True,
        seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        g = torch.Generator().manual_seed(seed)
        cube = torch.rand((batch_size, height, width, channels), generator=g, dtype=torch.float32)
        mask: torch.Tensor | None = None
        if with_mask:
            mask = torch.randint(
                low=0,
                high=4,
                size=(batch_size, height, width),
                generator=g,
                dtype=torch.int32,
            )
        return cube, mask

    return _make
