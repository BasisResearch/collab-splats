import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from collab_splats.semantics.segmentation import (
    BaseSegmentation,
    MobileSAMSegmentation,
    SAM3Segmentation,
    create_composite_mask,
    mobile_sam,
)
from collab_splats.semantics.segmentation.base import convert_matched_mask



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


def test_create_composite_mask_empty_results_raises():
    """No results means no shape; an empty (0, 0) mask would break every caller downstream."""
    with pytest.raises(ValueError, match="no results"):
        create_composite_mask([])


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


def test_create_composite_mask_min_visible_frac():
    """The buried-mask floor is a kwarg: at 0.05, a 9.1%-visible mask survives."""
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
    assert create_composite_mask(results)[0, 10] == 0
    assert create_composite_mask(results, min_visible_frac=0.05)[0, 10] != 0


def test_convert_matched_mask_label_count_mismatch_raises():
    masks = np.array([[1, 2], [2, 0]])
    with pytest.raises(ValueError, match="labels"):
        convert_matched_mask(torch.tensor([0]), masks)


########################################################
########## BaseSegmentation registry ##################
########################################################


def test_registry_get_mobilesamv2():
    cls = BaseSegmentation.get("mobilesamv2")
    assert cls is MobileSAMSegmentation


def test_registry_unknown_raises():
    with pytest.raises((KeyError, ValueError)):
        BaseSegmentation.get("nonexistent-backend")


def test_mobilesamv2_segment_with_text_raises():
    seg = MobileSAMSegmentation.__new__(MobileSAMSegmentation)
    seg.seg_model = MagicMock()
    seg.object_model = MagicMock()
    seg.predictor = MagicMock()
    seg.strategy = "object"
    with pytest.raises(NotImplementedError, match="does not support text-prompted"):
        seg.segment_with_text(MagicMock(), "a dog")


def test_base_segmentation_abstract():
    with pytest.raises(TypeError):
        BaseSegmentation()


def _mobilesam_stub(strategy="object", box_batch_size=320):
    with patch.object(mobile_sam, "_load_mobile_sam", return_value=(MagicMock(), MagicMock(), MagicMock())):
        return mobile_sam.MobileSAMSegmentation(strategy=strategy, box_batch_size=box_batch_size)


def test_mobilesamv2_bad_strategy_raises_at_init():
    with pytest.raises(ValueError, match="Strategy 'grid'"):
        _mobilesam_stub(strategy="grid")


def test_mobilesamv2_no_detections_returns_empty_stack():
    seg = _mobilesam_stub()
    seg.object_model.return_value = []
    masks, results = seg.segment(np.zeros((5, 7, 3), dtype=np.uint8))
    assert masks.shape == (0, 5, 7)
    assert masks.dtype == torch.float32
    assert results == []


def test_mobilesamv2_auto_empty_returns_empty_stack(monkeypatch):
    gen = MagicMock()
    gen.return_value.generate.return_value = []
    monkeypatch.setattr(mobile_sam, "SamAutomaticMaskGenerator", gen)
    seg = _mobilesam_stub(strategy="auto")
    masks, results = seg.segment(np.zeros((5, 7, 3), dtype=np.uint8))
    assert masks.shape == (0, 5, 7)
    assert results == []


def test_mobilesamv2_box_batch_size_reaches_decoder_loop(monkeypatch):
    """Both batch_iterator calls get box_batch_size; detections with no masks give an empty stack."""
    # Two detections reach the loop; the spy records batch sizes and yields no batches
    seg = _mobilesam_stub(box_batch_size=16)
    det = MagicMock()
    det.boxes.__len__.return_value = 2
    seg.object_model.return_value = [det]
    seg.predictor.transform.apply_boxes.return_value = np.zeros((2, 4), dtype=np.float32)
    seg.seg_model.parameters.return_value = iter([torch.zeros(1)])
    sizes = []

    def spy(bs, *args):
        sizes.append(bs)
        return iter(())

    monkeypatch.setattr(mobile_sam, "batch_iterator", spy)

    masks, results = seg.segment(np.zeros((5, 7, 3), dtype=np.uint8))

    assert sizes == [16, 16]
    assert masks.shape == (0, 5, 7)
    assert results == []


########################################################
########## SAM3Segmentation ###########################
########################################################

def test_registry_get_sam3():
    cls = BaseSegmentation.get("sam3")
    assert cls is SAM3Segmentation


def test_sam3_segment_with_text_interface():
    """segment_with_text exists and calls SAM3 processor — mocked to avoid loading model."""
    mock_masks = torch.zeros(2, 1, 4, 4)
    mock_boxes = torch.zeros(2, 4)
    mock_scores = torch.ones(2)
    mock_output = {"masks": mock_masks, "boxes": mock_boxes, "scores": mock_scores}

    mock_processor = MagicMock()
    mock_processor.set_image.return_value = "state"
    mock_processor.set_text_prompt.return_value = mock_output

    builder = types.ModuleType("sam3.model_builder")
    builder.build_sam3_image_model = MagicMock()
    proc = types.ModuleType("sam3.model.sam3_image_processor")
    proc.Sam3Processor = MagicMock(return_value=mock_processor)
    fake = {
        "sam3": types.ModuleType("sam3"),
        "sam3.model": types.ModuleType("sam3.model"),
        "sam3.model_builder": builder,
        "sam3.model.sam3_image_processor": proc,
    }
    with patch.dict(sys.modules, fake):
        seg = SAM3Segmentation(confidence_threshold=0.5)

    fake_img = Image.new("RGB", (4, 4))
    masks, boxes, scores = seg.segment_with_text(fake_img, "a cat")

    mock_processor.set_image.assert_called_once_with(fake_img)
    mock_processor.set_text_prompt.assert_called_once_with(state="state", prompt="a cat")
    assert masks.shape == (2, 1, 4, 4)
    assert boxes.shape == (2, 4)
    assert scores.shape == (2,)


def test_sam3_missing_dep_names_install():
    with patch.dict(sys.modules, {"sam3": None}):
        with pytest.raises(ImportError, match="huggingface-cli login"):
            SAM3Segmentation()


def test_sam3_broken_transitive_dep_propagates():
    """A missing dependency of sam3 surfaces under its own name, not as 'sam3 is not installed'."""
    # The processor module "loads" but its body needs a missing third-party package
    proc = types.ModuleType("sam3.model.sam3_image_processor")

    def missing(name):
        raise ModuleNotFoundError("No module named 'timm'", name="timm")

    proc.__getattr__ = missing
    fake = {
        "sam3": types.ModuleType("sam3"),
        "sam3.model": types.ModuleType("sam3.model"),
        "sam3.model.sam3_image_processor": proc,
    }
    with patch.dict(sys.modules, fake):
        with pytest.raises(ModuleNotFoundError, match="timm") as exc:
            SAM3Segmentation()
    assert "huggingface-cli" not in str(exc.value)
