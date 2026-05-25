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


def test_create_composite_mask_empty_results():
    """create_composite_mask must not crash when no masks pass confidence threshold."""
    from collab_splats.semantics.segmentation import create_composite_mask
    import numpy as np
    results = [{"predicted_iou": 0.5, "segmentation": np.zeros((4, 4), dtype=np.uint8)}]
    result = create_composite_mask(results, confidence_threshold=0.99)
    assert isinstance(result, np.ndarray)


########################################################
########## BaseSegmentation registry ##################
########################################################


def test_registry_get_mobilesamv2():
    from collab_splats.semantics.segmentation import BaseSegmentation, MobileSAMSegmentation
    cls = BaseSegmentation.get("mobilesamv2")
    assert cls is MobileSAMSegmentation


def test_registry_unknown_raises():
    from collab_splats.semantics.segmentation import BaseSegmentation
    with pytest.raises((KeyError, ValueError)):
        BaseSegmentation.get("nonexistent-backend")


def test_mobilesamv2_segment_with_text_raises():
    from collab_splats.semantics.segmentation import MobileSAMSegmentation
    from unittest.mock import MagicMock
    seg = MobileSAMSegmentation.__new__(MobileSAMSegmentation)
    seg.seg_model = MagicMock()
    seg.object_model = MagicMock()
    seg.predictor = MagicMock()
    seg.strategy = "object"
    with pytest.raises(NotImplementedError, match="does not support text-prompted"):
        seg.segment_with_text(MagicMock(), "a dog")


def test_base_segmentation_abstract():
    from collab_splats.semantics.segmentation import BaseSegmentation
    with pytest.raises(TypeError):
        BaseSegmentation()


########################################################
########## SAM3Segmentation ###########################
########################################################

def test_registry_get_sam3():
    from collab_splats.semantics.segmentation import BaseSegmentation, SAM3Segmentation
    cls = BaseSegmentation.get("sam3")
    assert cls is SAM3Segmentation


def test_sam3_segment_with_text_interface():
    """segment_with_text exists and calls SAM3 processor — mocked to avoid loading model."""
    from collab_splats.semantics.segmentation import SAM3Segmentation
    from unittest.mock import patch, MagicMock
    import torch

    mock_masks = torch.zeros(2, 1, 4, 4)
    mock_boxes = torch.zeros(2, 4)
    mock_scores = torch.ones(2)
    mock_output = {"masks": mock_masks, "boxes": mock_boxes, "scores": mock_scores}

    mock_processor = MagicMock()
    mock_processor.set_image.return_value = "state"
    mock_processor.set_text_prompt.return_value = mock_output

    with patch("collab_splats.semantics.segmentation.sam3.build_sam3_image_model",
               return_value=MagicMock()):
        with patch("collab_splats.semantics.segmentation.sam3.Sam3Processor",
                   return_value=mock_processor):
            seg = SAM3Segmentation(confidence_threshold=0.5)

    from PIL import Image
    fake_img = Image.new("RGB", (4, 4))
    masks, boxes, scores = seg.segment_with_text(fake_img, "a cat")

    mock_processor.set_image.assert_called_once_with(fake_img)
    mock_processor.set_text_prompt.assert_called_once_with(state="state", prompt="a cat")
    assert masks.shape == (2, 1, 4, 4)
    assert boxes.shape == (2, 4)
    assert scores.shape == (2,)
