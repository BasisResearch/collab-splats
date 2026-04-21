import numpy as np
import pytest
from collab_splats.semantics.segmentation import create_composite_mask


def _make_results(n: int, iou: float = 0.9):
    mask = np.ones((8, 8), dtype=np.uint8)
    return [{"segmentation": mask, "predicted_iou": iou} for _ in range(n)]


def test_create_composite_mask_no_stdout(capsys):
    results = _make_results(3)
    create_composite_mask(results)
    captured = capsys.readouterr()
    assert captured.out == "", f"Unexpected stdout: {captured.out!r}"
