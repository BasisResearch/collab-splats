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


def test_create_composite_mask_ids_ascend_with_confidence():
    """Disjoint masks are numbered 1..N in ascending order of predicted_iou."""
    # Three disjoint one-column stripes in a 3x4 image, confidence rising with column index
    results = []
    for col, iou in zip((0, 1, 2), (0.90, 0.95, 0.99)):
        seg = np.zeros((3, 4), dtype=bool)
        seg[:, col] = True
        results.append({"segmentation": seg, "predicted_iou": iou})

    composite = create_composite_mask(results)

    # Column 3 is background; the stripes carry 1, 2, 3 in confidence order
    assert composite[0].tolist() == [1, 2, 3, 0]


def test_create_composite_mask_highest_confidence_wins_overlap():
    """Where two masks overlap, the higher-confidence one owns the shared pixels."""
    # Low confidence covers columns 0-1, high covers columns 1-2, so column 1 is contested
    low = np.zeros((3, 4), dtype=bool)
    low[:, 0:2] = True
    high = np.zeros((3, 4), dtype=bool)
    high[:, 1:3] = True
    results = [
        {"segmentation": low, "predicted_iou": 0.90},
        {"segmentation": high, "predicted_iou": 0.99},
    ]

    composite = create_composite_mask(results)

    # The contested column joins the high-confidence mask's exclusive column, not the low one's
    assert composite[0, 1] == composite[0, 2]
    assert composite[0, 0] != composite[0, 1]
    assert composite[0, 0] != 0


def test_create_composite_mask_area_ratio_drops_buried_mask():
    """The >10% area floor divides surviving pixels by the mask those pixels came from."""
    # Input order is deliberately not confidence order, so masks[] order and the paint order
    # disagree. `wide` (11 cols) is all but buried by `medium` (10 cols); `small` sits apart.
    small = np.zeros((3, 13), dtype=bool)
    small[:, 12] = True
    wide = np.zeros((3, 13), dtype=bool)
    wide[:, 0:11] = True
    medium = np.zeros((3, 13), dtype=bool)
    medium[:, 0:10] = True
    results = [
        {"segmentation": small, "predicted_iou": 0.99},
        {"segmentation": wide, "predicted_iou": 0.90},
        {"segmentation": medium, "predicted_iou": 0.95},
    ]

    composite = create_composite_mask(results)

    # `wide` keeps 3 of its own 33 pixels (9.1%), under the floor, so it is dropped entirely
    assert composite[0, 10] == 0
    # `medium` and `small` keep all of theirs and stay numbered in confidence order
    assert composite[0, 0:10].tolist() == [2] * 10
    assert composite[0, 12] == 3


def test_create_composite_mask_truly_empty_results():
    """An empty results list has no shape to borrow, so it returns an empty (0, 0) mask."""
    # Distinct from the threshold-filtered case above: that path still has results[0] to
    # take a shape from, this one has nothing at all.
    result = create_composite_mask([])

    assert result.shape == (0, 0)
    assert result.dtype == np.uint16


def test_create_composite_mask_more_than_255_masks():
    """Mask IDs past 255 must survive — uint8 raised OverflowError on the 256th."""
    # 300 disjoint one-row stripes, all above threshold, so every one of them earns an ID.
    # SAM's 32x32 default point grid routinely clears this many proposals.
    results = []
    for row in range(300):
        seg = np.zeros((300, 4), dtype=bool)
        seg[row, :] = True
        results.append({"segmentation": seg, "predicted_iou": 0.9})

    composite = create_composite_mask(results)

    assert composite.dtype == np.uint16
    assert composite.max() == 300
    assert np.array_equal(np.unique(composite), np.arange(1, 301, dtype=np.uint16))


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
