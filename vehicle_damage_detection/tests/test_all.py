"""
Unit Tests for Vehicle Damage Detection System

Tests:
  - Model forward pass
  - Loss functions
  - Metrics computation
  - Post-processing pipeline
  - Preprocessing utilities

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short
"""

import sys
import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.losses import (
    focal_loss, smooth_l1_loss, dice_loss,
    mask_bce_loss, severity_loss, VehicleDamageLoss
)
from src.evaluation.metrics import (
    DetectionMetrics, SegmentationMetrics, SeverityMetrics,
    compute_iou_matrix, compute_mask_iou
)
from src.utils.postprocessing import (
    non_maximum_suppression, get_severity_band,
    estimate_repair_cost, build_damage_report
)
from src.data.preprocessing import (
    letterbox_resize, suppress_specular_highlights,
    apply_clahe, normalize, denormalize
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_image():
    """Random BGR image for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_boxes():
    """Random valid bounding boxes."""
    return np.array([
        [10, 20, 100, 150],
        [200, 100, 400, 350],
        [50, 300, 200, 450],
    ], dtype=np.float32)


@pytest.fixture
def min_cfg():
    """Minimal config dict for testing."""
    return {
        'model': {'num_classes': 6, 'num_anchors': 9,
                  'fpn_out_channels': 256, 'freeze_backbone_layers': 0,
                  'image_size': [64, 64]},
        'training': {
            'loss_weights': {'cls': 1.0, 'box': 1.0, 'mask': 2.0,
                             'dice': 0.5, 'severity': 1.5},
            'class_weights': {
                'background': 0.1, 'scratch': 2.0, 'dent': 1.5,
                'glass_damage': 1.8, 'tear': 1.6, 'body_deform': 1.2
            }
        },
        'classes': ['background', 'scratch', 'dent',
                    'glass_damage', 'tear', 'body_deform']
    }


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_letterbox_resize_output_shape(self, dummy_image):
        result, meta = letterbox_resize(dummy_image, target=(640, 640))
        assert result.shape == (640, 640, 3), \
            f"Expected (640,640,3), got {result.shape}"

    def test_letterbox_resize_aspect_ratio_preserved(self):
        """For a square image, scale should be uniform."""
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        result, meta = letterbox_resize(img, target=(640, 640))
        assert meta['scale'] == pytest.approx(2.0)

    def test_letterbox_resize_meta_keys(self, dummy_image):
        _, meta = letterbox_resize(dummy_image)
        for key in ['scale', 'pad_top', 'pad_left', 'orig_h', 'orig_w']:
            assert key in meta

    def test_specular_suppression_no_highlights(self, dummy_image):
        """Image with no highlights should be returned unchanged."""
        dark_image = np.clip(dummy_image, 0, 100).astype(np.uint8)
        result = suppress_specular_highlights(dark_image, threshold=240)
        assert result.shape == dark_image.shape

    def test_specular_suppression_with_highlights(self):
        """Image with white region should have highlights suppressed."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = 255  # white patch
        result = suppress_specular_highlights(img, threshold=200)
        assert result.shape == img.shape
        # The white patch should be partially suppressed
        assert result[50, 50].mean() < 255

    def test_clahe_output_shape(self, dummy_image):
        result = apply_clahe(dummy_image)
        assert result.shape == dummy_image.shape

    def test_normalize_range(self, dummy_image):
        normalized = normalize(dummy_image)
        assert normalized.dtype == np.float32
        # Values should be roughly in [-3, 3] after ImageNet normalization
        assert normalized.min() > -5.0
        assert normalized.max() < 5.0

    def test_normalize_denormalize_roundtrip(self, dummy_image):
        """Normalize then denormalize should approximately recover original."""
        normalized = normalize(dummy_image)
        recovered = denormalize(normalized)
        assert recovered.shape == dummy_image.shape
        # Allow some precision loss
        diff = np.abs(dummy_image.astype(np.float32) -
                      recovered.astype(np.float32))
        assert diff.mean() < 5.0  # within 5 pixel units average


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestLossFunctions:

    def test_focal_loss_perfect_prediction(self):
        """Perfect predictions should yield near-zero focal loss."""
        y_true = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.999, 0.001, 0.001]], dtype=tf.float32)
        loss = focal_loss(y_true, y_pred)
        assert float(loss) < 0.1

    def test_focal_loss_worst_prediction(self):
        """Worst predictions should yield higher focal loss than perfect."""
        y_true = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
        pred_perfect = tf.constant([[0.99, 0.005, 0.005]], dtype=tf.float32)
        pred_wrong = tf.constant([[0.01, 0.5, 0.49]], dtype=tf.float32)
        loss_perfect = focal_loss(y_true, pred_perfect)
        loss_wrong = focal_loss(y_true, pred_wrong)
        assert float(loss_wrong) > float(loss_perfect)

    def test_focal_loss_output_type(self):
        y_true = tf.ones([2, 10, 6], dtype=tf.float32) * 0.1
        y_pred = tf.ones([2, 10, 6], dtype=tf.float32) * 0.5
        loss = focal_loss(y_true, y_pred)
        assert loss.dtype in [tf.float32, tf.float16]

    def test_smooth_l1_loss_zero_error(self):
        """Same input should yield zero loss."""
        y = tf.constant([[1.0, 2.0, 3.0, 4.0]], dtype=tf.float32)
        loss = smooth_l1_loss(y, y)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_smooth_l1_loss_positive(self):
        y_true = tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.5, 0.5, 0.5, 0.5]], dtype=tf.float32)
        loss = smooth_l1_loss(y_true, y_pred)
        assert float(loss) > 0.0

    def test_dice_loss_perfect_prediction(self):
        """Perfect mask prediction should yield near-zero dice loss."""
        y = tf.ones([2, 4, 4, 1], dtype=tf.float32)
        loss = dice_loss(y, y)
        assert float(loss) < 0.01

    def test_dice_loss_no_overlap(self):
        """Non-overlapping masks should yield dice loss near 1.0."""
        y_true = tf.constant([[[[1.0], [0.0]], [[0.0], [0.0]]]], dtype=tf.float32)
        y_pred = tf.constant([[[[0.0], [1.0]], [[0.0], [0.0]]]], dtype=tf.float32)
        loss = dice_loss(y_true, y_pred)
        assert float(loss) > 0.5

    def test_severity_loss_positive(self):
        y_true = tf.constant([[0.7]], dtype=tf.float32)
        y_pred = tf.constant([[0.3]], dtype=tf.float32)
        loss = severity_loss(y_true, y_pred)
        assert float(loss) > 0.0

    def test_severity_loss_zero_for_same_input(self):
        y = tf.constant([[0.5]], dtype=tf.float32)
        loss = severity_loss(y, y)
        assert float(loss) == pytest.approx(0.0, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_iou_matrix_perfect_overlap(self, dummy_boxes):
        """Same boxes should have IoU = 1.0 on diagonal."""
        iou = compute_iou_matrix(dummy_boxes, dummy_boxes)
        assert iou.shape == (3, 3)
        np.testing.assert_allclose(np.diag(iou), 1.0, atol=1e-5)

    def test_iou_matrix_no_overlap(self):
        """Non-overlapping boxes should have IoU = 0.0."""
        boxes_a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes_b = np.array([[100, 100, 200, 200]], dtype=np.float32)
        iou = compute_iou_matrix(boxes_a, boxes_b)
        assert iou[0, 0] == pytest.approx(0.0)

    def test_mask_iou_perfect(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        assert compute_mask_iou(mask, mask) == pytest.approx(1.0)

    def test_mask_iou_no_overlap(self):
        mask_a = np.zeros((100, 100), dtype=np.uint8)
        mask_b = np.zeros((100, 100), dtype=np.uint8)
        mask_a[:50] = 1
        mask_b[50:] = 1
        assert compute_mask_iou(mask_a, mask_b) == pytest.approx(0.0)

    def test_detection_metrics_empty(self):
        """Empty state should return zero metrics."""
        metrics = DetectionMetrics(num_classes=6)
        results = metrics.compute()
        assert results['map_50'] == 0.0

    def test_severity_metrics_perfect(self):
        """Perfect predictions should yield zero MAE."""
        metrics = SeverityMetrics()
        y = np.array([[0.3], [0.6], [0.9]])
        metrics.update(y, y)
        results = metrics.compute()
        assert results['severity_mae'] == pytest.approx(0.0, abs=1e-6)
        assert results['severity_band_acc'] == pytest.approx(1.0)

    def test_severity_metrics_band_accuracy(self):
        """Test severity band classification."""
        metrics = SeverityMetrics()
        y_true = np.array([[0.1], [0.4], [0.8]])  # minor, moderate, severe
        y_pred = np.array([[0.2], [0.5], [0.7]])  # same bands
        metrics.update(y_true, y_pred)
        results = metrics.compute()
        assert results['severity_band_acc'] == pytest.approx(1.0)

    def test_segmentation_metrics_perfect(self):
        """Perfect mask predictions should yield mIoU = 1.0."""
        metrics = SegmentationMetrics(num_classes=2)
        masks = np.random.randint(0, 2, (2, 32, 32, 2)).astype(np.float32)
        metrics.update(masks, masks)
        results = metrics.compute()
        assert results['miou'] == pytest.approx(1.0, abs=0.01)
        assert results['pixel_accuracy'] == pytest.approx(1.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPostProcessing:

    def test_nms_removes_duplicates(self):
        """Highly overlapping boxes with same class should be reduced."""
        boxes = np.array([
            [10, 10, 100, 100],
            [12, 12, 102, 102],  # near-duplicate
            [200, 200, 300, 300],  # separate box
        ], dtype=np.float32)
        scores = np.array([0.9, 0.85, 0.8], dtype=np.float32)
        classes = np.array([1, 1, 2], dtype=np.int32)

        filtered_boxes, filtered_scores, filtered_classes = \
            non_maximum_suppression(boxes, scores, classes,
                                    iou_threshold=0.5,
                                    score_threshold=0.5)

        # Should keep at most 2 (one per non-overlapping group)
        assert len(filtered_boxes) <= 2

    def test_nms_below_threshold_filtered(self):
        """Boxes below score threshold should be removed."""
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        scores = np.array([0.2], dtype=np.float32)
        classes = np.array([1], dtype=np.int32)

        filtered_boxes, _, _ = non_maximum_suppression(
            boxes, scores, classes, score_threshold=0.45
        )
        assert len(filtered_boxes) == 0

    def test_severity_band_thresholds(self):
        assert get_severity_band(0.0) == (0, 'MINOR')
        assert get_severity_band(0.15) == (0, 'MINOR')
        assert get_severity_band(0.29) == (0, 'MINOR')
        assert get_severity_band(0.30) == (1, 'MODERATE')
        assert get_severity_band(0.50) == (1, 'MODERATE')
        assert get_severity_band(0.64) == (1, 'MODERATE')
        assert get_severity_band(0.65) == (2, 'SEVERE')
        assert get_severity_band(1.00) == (2, 'SEVERE')

    def test_repair_cost_ranges(self):
        lo, hi = estimate_repair_cost('scratch', 'MINOR')
        assert lo > 0
        assert hi > lo

        lo_sev, hi_sev = estimate_repair_cost('scratch', 'SEVERE')
        lo_min, hi_min = estimate_repair_cost('scratch', 'MINOR')
        assert lo_sev > lo_min  # severe should cost more

    def test_damage_report_structure(self):
        boxes = np.array([[10, 10, 100, 100], [200, 200, 350, 350]],
                          dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        classes = np.array([1, 2], dtype=np.int32)
        masks = [np.ones((480, 640), dtype=np.uint8) * 0,
                 np.ones((480, 640), dtype=np.uint8) * 0]
        masks[0][20:80, 20:80] = 1
        masks[1][210:330, 210:330] = 1
        severities = np.array([0.35, 0.6], dtype=np.float32)
        occlusions = np.array([0.0, 0.1], dtype=np.float32)

        report = build_damage_report(
            boxes, scores, classes, masks, severities, occlusions,
            480, 640
        )

        assert 'vehicle_overall_severity' in report
        assert 'vehicle_overall_label' in report
        assert 'damage_count' in report
        assert 'regions' in report
        assert 'cost_estimate_usd' in report
        assert report['damage_count'] == 2
        assert report['vehicle_overall_label'] in ['MINOR', 'MODERATE', 'SEVERE']
        assert report['cost_estimate_usd']['low'] > 0

    def test_empty_detections_report(self):
        """Empty detections should yield a valid zero-damage report."""
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
        classes = np.zeros(0, dtype=np.int32)

        report = build_damage_report(
            boxes, scores, classes, [], np.array([]), np.array([]),
            480, 640
        )
        assert report['damage_count'] == 0
        assert report['vehicle_overall_severity'] == 0.0
        assert report['cost_estimate_usd']['midpoint'] == 0


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestModelIntegration:
    """Integration test: build model and run one forward pass."""

    @pytest.mark.slow
    def test_model_forward_pass_shape(self, min_cfg):
        """Model should return outputs with expected shapes."""
        from src.model import build_model
        model = build_model(min_cfg)
        dummy_input = tf.zeros([2, 64, 64, 3])
        output = model(dummy_input, training=False)

        assert 'cls' in output
        assert 'box' in output
        assert 'masks' in output
        assert 'severity' in output
        assert 'occlusion' in output

        # Severity shape
        assert output['severity'].shape == (2, 1)
        assert output['occlusion'].shape == (2, 1)

    @pytest.mark.slow
    def test_model_severity_range(self, min_cfg):
        """Severity output should be in [0, 1] (sigmoid)."""
        from src.model import build_model
        model = build_model(min_cfg)
        dummy_input = tf.zeros([1, 64, 64, 3])
        output = model(dummy_input, training=False)

        sev = output['severity'].numpy()
        assert sev.min() >= 0.0
        assert sev.max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
