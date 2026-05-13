"""Contract checks for AugmentationCompose's published port specs."""

from __future__ import annotations

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage

from cuvis_ai_augment.node.compose import AugmentationCompose


def test_compose_is_node_subclass() -> None:
    assert issubclass(AugmentationCompose, Node)


def test_input_specs_contract() -> None:
    ins = AugmentationCompose.INPUT_SPECS
    assert set(ins.keys()) == {"cube", "mask"}
    assert ins["cube"].dtype == torch.float32
    assert ins["cube"].shape == (-1, -1, -1, -1)
    assert ins["mask"].shape == (-1, -1, -1)
    assert ins["mask"].optional is True


def test_output_specs_contract() -> None:
    outs = AugmentationCompose.OUTPUT_SPECS
    assert set(outs.keys()) == {"cube", "mask"}
    assert outs["cube"].dtype == torch.float32
    assert outs["cube"].shape == (-1, -1, -1, -1)
    assert outs["mask"].shape == (-1, -1, -1)
    assert outs["mask"].optional is True


def test_execution_stages_train_only() -> None:
    assert AugmentationCompose.execution_stages == {ExecutionStage.TRAIN}


def test_instantiable_with_empty_transforms() -> None:
    node = AugmentationCompose(transforms=[])
    assert hasattr(node, "forward")
