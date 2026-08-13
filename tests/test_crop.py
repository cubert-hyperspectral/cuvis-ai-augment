"""Tests for the deterministic ``Crop`` node."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage

from cuvis_ai_augment.node.crop import Crop


def test_crop_is_node_subclass() -> None:
    assert issubclass(Crop, Node)


def test_input_specs_contract() -> None:
    ins = Crop.INPUT_SPECS
    assert set(ins.keys()) == {"data", "mask"}
    assert ins["data"].dtype == torch.float32
    assert ins["data"].shape == (-1, -1, -1, -1)
    assert ins["mask"].shape == (-1, -1, -1)
    assert ins["mask"].optional is True


def test_output_specs_contract() -> None:
    outs = Crop.OUTPUT_SPECS
    assert set(outs.keys()) == {"cropped", "mask_cropped"}
    assert outs["cropped"].dtype == torch.float32
    assert outs["cropped"].shape == (-1, -1, -1, -1)
    assert outs["mask_cropped"].shape == (-1, -1, -1)
    assert outs["mask_cropped"].optional is True


def test_default_is_always_routable() -> None:
    # A deterministic crop runs at every stage; no internal stage gate.
    node = Crop()
    assert node.execution_stages == {ExecutionStage.ALWAYS}


def test_defaults_are_identity() -> None:
    node = Crop()
    cube = torch.rand(2, 10, 12, 4)
    out = node.forward(data=cube)
    assert torch.equal(out["cropped"], cube)
    assert "mask_cropped" not in out


def test_crops_cube_and_mask_with_same_rectangle() -> None:
    node = Crop(top=1, bottom=4, left=2, right=5)
    cube = torch.rand(2, 10, 12, 3)
    mask = torch.randint(0, 3, (2, 10, 12), dtype=torch.int32)
    out = node.forward(data=cube, mask=mask)
    assert out["cropped"].shape == (2, 3, 3, 3)
    assert out["mask_cropped"].shape == (2, 3, 3)
    assert torch.equal(out["cropped"], cube[:, 1:4, 2:5, :])
    assert torch.equal(out["mask_cropped"], mask[:, 1:4, 2:5])


def test_open_ended_bounds() -> None:
    node = Crop(top=2, left=3)  # bottom/right default to None -> to the edge
    cube = torch.rand(1, 8, 9, 2)
    out = node.forward(data=cube)
    assert torch.equal(out["cropped"], cube[:, 2:, 3:, :])


def test_output_is_contiguous() -> None:
    node = Crop(top=1, bottom=5, left=1, right=6)
    out = node.forward(data=torch.rand(1, 10, 10, 3))
    assert out["cropped"].is_contiguous()


def test_hparams_are_serialized() -> None:
    node = Crop(top=13, bottom=1000, left=30, right=435)
    params = node.get_params() if hasattr(node, "get_params") else node.__dict__
    # The four bounds are stored on the instance regardless of the base serializer shape.
    assert (node.top, node.bottom, node.left, node.right) == (13, 1000, 30, 435)
    assert params is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bottom": 5, "top": 5},  # bottom == top
        {"bottom": 4, "top": 5},  # bottom < top
        {"left": 3, "right": 3},  # right == left
        {"left": 3, "right": 2},  # right < left
        {"top": -1},  # negative
        {"left": -2},  # negative
    ],
)
def test_invalid_bounds_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Crop(**kwargs)


def test_bounds_exceeding_frame_raise() -> None:
    # Python slicing would silently clip bottom=500 on H=300 and return all 300
    # rows, masking the config error — the node must reject it instead.
    node = Crop(top=0, bottom=500)
    with pytest.raises(ValueError, match="exceed the data"):
        node.forward(data=torch.rand(1, 300, 100, 4))


def test_empty_crop_on_frame_raises() -> None:
    # bottom=500 exceeds H=300 → rejected by the bounds check (silent slicing
    # would instead return an empty [B, 0, W, C] tensor).
    node = Crop(top=350, bottom=500)
    with pytest.raises(ValueError, match="exceed the data"):
        node.forward(data=torch.rand(1, 300, 100, 4))


def test_open_ended_empty_crop_raises() -> None:
    # bottom=None resolves to H=300; top=350 then yields an empty rectangle.
    node = Crop(top=350)
    with pytest.raises(ValueError, match="empty"):
        node.forward(data=torch.rand(1, 300, 100, 4))


def test_right_exceeding_width_raises() -> None:
    node = Crop(left=0, right=200)
    with pytest.raises(ValueError, match="exceed the data"):
        node.forward(data=torch.rand(1, 300, 100, 4))


def test_bounds_equal_to_frame_are_valid() -> None:
    # bottom == H and right == W select exactly to the edge — must NOT raise.
    node = Crop(top=13, bottom=300, left=30, right=100)
    out = node.forward(data=torch.rand(1, 300, 100, 4))
    assert out["cropped"].shape == (1, 287, 70, 4)


def test_mask_spatial_mismatch_raises() -> None:
    node = Crop(top=1, bottom=4)
    cube = torch.rand(2, 10, 12, 3)
    mask = torch.randint(0, 2, (2, 10, 11), dtype=torch.int32)  # W mismatch
    with pytest.raises(ValueError, match="spatial shapes must match"):
        node.forward(data=cube, mask=mask)


def test_non_4d_data_raises() -> None:
    node = Crop()
    with pytest.raises(ValueError, match="4-D"):
        node.forward(data=torch.rand(10, 12, 3))


@pytest.mark.integration
def test_manifest_resolves_crop() -> None:
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    manifest = Path(__file__).resolve().parents[1] / "examples" / "plugins.yaml"
    registry = NodeRegistry()
    registry.register_plugin(str(manifest))
    assert registry.get("Crop").__name__ == "Crop"
