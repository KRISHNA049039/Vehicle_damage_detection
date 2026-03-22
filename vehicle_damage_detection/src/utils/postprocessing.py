"""
Post-Processing Pipeline for Vehicle Damage Detection

Includes:
  - Anchor-based box decoding
  - Multi-class NMS
  - Damage report generation
  - Damage cost estimation (heuristic)
"""

import numpy as np
import tensorflow as tf
import cv2
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ['background', 'scratch', 'dent',
               'glass_damage', 'tear', 'body_deform']

SEVERITY_LABELS = {
    0: 'MINOR',
    1: 'MODERATE',
    2: 'SEVERE'
}

COLORS = {
    'scratch':      (255, 0, 128),
    'dent':         (0, 128, 255),
    'glass_damage': (0, 255, 200),
    'tear':         (255, 128, 0),
    'body_deform':  (200, 0, 255)
}

# Heuristic repair cost ranges (USD) per damage type × severity
COST_RANGES = {
    'scratch':      {'MINOR': (50, 200),    'MODERATE': (200, 800),   'SEVERE': (800, 2000)},
    'dent':         {'MINOR': (100, 400),   'MODERATE': (400, 1500),  'SEVERE': (1500, 4000)},
    'glass_damage': {'MINOR': (100, 300),   'MODERATE': (300, 800),   'SEVERE': (800, 3000)},
    'tear':         {'MINOR': (150, 500),   'MODERATE': (500, 1800),  'SEVERE': (1800, 5000)},
    'body_deform':  {'MINOR': (300, 1000),  'MODERATE': (1000, 4000), 'SEVERE': (4000, 12000)}
}


# ─────────────────────────────────────────────────────────────────────────────
# BOX DECODING
# ─────────────────────────────────────────────────────────────────────────────

def decode_boxes(box_deltas: np.ndarray,
                 anchors: np.ndarray,
                 weights: Tuple[float, ...] = (10., 10., 5., 5.)) -> np.ndarray:
    """
    Decode predicted box deltas relative to anchors.

    Encoding convention:
        tx = (x - xa) / wa * wx
        ty = (y - ya) / ha * wy
        tw = log(w / wa) * ww
        th = log(h / ha) * wh

    Args:
        box_deltas: [N, 4] predicted (dx, dy, dw, dh)
        anchors:    [N, 4] anchor boxes [x1, y1, x2, y2]
        weights:    scaling weights for (x, y, w, h)
    Returns:
        boxes: [N, 4] decoded boxes [x1, y1, x2, y2]
    """
    wx, wy, ww, wh = weights

    # Convert anchors to center format
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + 0.5 * aw
    ay = anchors[:, 1] + 0.5 * ah

    # Apply deltas
    dx = box_deltas[:, 0] / wx
    dy = box_deltas[:, 1] / wy
    dw = box_deltas[:, 2] / ww
    dh = box_deltas[:, 3] / wh

    # Clamp to prevent exp overflow
    dw = np.clip(dw, -4, 4)
    dh = np.clip(dh, -4, 4)

    pred_cx = dx * aw + ax
    pred_cy = dy * ah + ay
    pred_w = np.exp(dw) * aw
    pred_h = np.exp(dh) * ah

    x1 = pred_cx - 0.5 * pred_w
    y1 = pred_cy - 0.5 * pred_h
    x2 = pred_cx + 0.5 * pred_w
    y2 = pred_cy + 0.5 * pred_h

    return np.stack([x1, y1, x2, y2], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# NON-MAXIMUM SUPPRESSION
# ─────────────────────────────────────────────────────────────────────────────

def non_maximum_suppression(boxes: np.ndarray,
                             scores: np.ndarray,
                             classes: np.ndarray,
                             iou_threshold: float = 0.5,
                             score_threshold: float = 0.45,
                             max_detections: int = 100) -> Tuple:
    """
    Class-aware NMS to remove redundant detections.

    Performs NMS per class independently to avoid suppressing
    overlapping detections of different damage types.

    Args:
        boxes:           [N, 4] in [x1, y1, x2, y2]
        scores:          [N] confidence scores
        classes:         [N] class indices
        iou_threshold:   suppress if IoU > this
        score_threshold: keep detections with score > this
        max_detections:  max number of final detections
    Returns:
        filtered boxes, scores, classes as arrays
    """
    # Score filtering
    keep_mask = scores > score_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    classes = classes[keep_mask]

    if len(boxes) == 0:
        return boxes, scores, classes

    # Use TensorFlow NMS for efficiency
    tf_boxes = tf.constant(boxes, dtype=tf.float32)
    tf_scores = tf.constant(scores, dtype=tf.float32)

    # Per-class NMS
    final_indices = []
    for cls_id in np.unique(classes):
        cls_mask = classes == cls_id
        cls_boxes = tf_boxes[cls_mask]
        cls_scores = tf_scores[cls_mask]
        cls_orig_idx = np.where(cls_mask)[0]

        if len(cls_boxes) == 0:
            continue

        nms_idx = tf.image.non_max_suppression(
            cls_boxes, cls_scores,
            max_output_size=max_detections,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        ).numpy()

        final_indices.extend(cls_orig_idx[nms_idx].tolist())

    if not final_indices:
        empty = np.zeros((0,), dtype=np.int32)
        return boxes[empty], scores[empty], classes[empty]

    final_indices = np.array(final_indices)
    # Sort by score
    score_order = np.argsort(-scores[final_indices])
    final_indices = final_indices[score_order[:max_detections]]

    return boxes[final_indices], scores[final_indices], classes[final_indices]


# ─────────────────────────────────────────────────────────────────────────────
# MASK POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def extract_instance_masks(mask_output: np.ndarray,
                            boxes: np.ndarray,
                            classes: np.ndarray,
                            orig_h: int, orig_w: int,
                            threshold: float = 0.5) -> List[np.ndarray]:
    """
    Extract per-instance binary masks from the mask prediction output.

    Args:
        mask_output: [H, W, num_classes] from model
        boxes:       [N, 4] detected boxes
        classes:     [N] class indices
        orig_h/w:    original image dimensions
        threshold:   binarization threshold
    Returns:
        List of [orig_h, orig_w] binary masks, one per detection
    """
    instance_masks = []
    mask_h, mask_w = mask_output.shape[:2]

    for box, cls in zip(boxes, classes):
        # Get class-specific probability map
        cls_mask = mask_output[:, :, int(cls)]

        # Crop to detection box (scaled to mask resolution)
        scale_x = mask_w / orig_w
        scale_y = mask_h / orig_h

        x1m = max(0, int(box[0] * scale_x))
        y1m = max(0, int(box[1] * scale_y))
        x2m = min(mask_w, int(box[2] * scale_x))
        y2m = min(mask_h, int(box[3] * scale_y))

        # Full mask (not just ROI crop for semantic-style mask head)
        full_mask_resized = cv2.resize(cls_mask, (orig_w, orig_h))
        binary = (full_mask_resized > threshold).astype(np.uint8)

        # Restrict to bounding box
        box_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        bx1, by1 = max(0, int(box[0])), max(0, int(box[1]))
        bx2, by2 = min(orig_w, int(box[2])), min(orig_h, int(box[3]))
        box_mask[by1:by2, bx1:bx2] = binary[by1:by2, bx1:bx2]

        instance_masks.append(box_mask)

    return instance_masks


# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY BANDING
# ─────────────────────────────────────────────────────────────────────────────

def get_severity_band(score: float) -> Tuple[int, str]:
    """Convert continuous severity score to discrete band."""
    if score < 0.30:
        return 0, 'MINOR'
    elif score < 0.65:
        return 1, 'MODERATE'
    else:
        return 2, 'SEVERE'


def estimate_repair_cost(damage_class: str,
                          severity_label: str,
                          num_regions: int = 1) -> Tuple[int, int]:
    """
    Heuristic repair cost estimation.

    Returns (low_estimate, high_estimate) in USD.
    """
    if damage_class not in COST_RANGES:
        return 0, 0
    lo, hi = COST_RANGES[damage_class][severity_label]
    return lo * num_regions, hi * num_regions


# ─────────────────────────────────────────────────────────────────────────────
# DAMAGE REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_damage_report(boxes: np.ndarray,
                         scores: np.ndarray,
                         classes: np.ndarray,
                         masks: List[np.ndarray],
                         severities: np.ndarray,
                         occlusions: np.ndarray,
                         orig_h: int,
                         orig_w: int) -> Dict:
    """
    Generate a structured damage report from detections.

    Args:
        boxes:      [N, 4] final detected boxes
        scores:     [N] confidence scores
        classes:    [N] class indices
        masks:      list of N binary masks [H, W]
        severities: [N] severity scores per detection
        occlusions: [N] occlusion estimates
        orig_h/w:   original image dimensions
    Returns:
        Structured damage report dict
    """
    total_pixels = orig_h * orig_w
    regions = []
    cost_lo_total, cost_hi_total = 0, 0

    for i, (box, score, cls_idx, mask, sev, occ) in enumerate(
        zip(boxes, scores, classes, masks, severities, occlusions)
    ):
        cls_idx = int(cls_idx)
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) \
            else f'class_{cls_idx}'

        if cls_name == 'background':
            continue

        area_px = int(mask.sum())
        area_pct = round(area_px / total_pixels * 100, 3)
        sev_band, sev_label = get_severity_band(float(sev))

        cost_lo, cost_hi = estimate_repair_cost(cls_name, sev_label)
        cost_lo_total += cost_lo
        cost_hi_total += cost_hi

        regions.append({
            'id': i,
            'class': cls_name,
            'confidence': round(float(score), 4),
            'bbox': {
                'x1': int(box[0]), 'y1': int(box[1]),
                'x2': int(box[2]), 'y2': int(box[3]),
                'width': int(box[2] - box[0]),
                'height': int(box[3] - box[1])
            },
            'mask_area_px': area_px,
            'mask_area_pct': area_pct,
            'severity_score': round(float(sev), 4),
            'severity_label': sev_label,
            'occlusion_level': round(float(occ), 3),
            'confidence_adjusted': round(
                float(score) * (1.0 - 0.3 * float(occ)), 4
            ),  # down-weight occluded detections
            'cost_estimate_usd': {
                'low': cost_lo, 'high': cost_hi
            }
        })

    # Vehicle-level aggregation
    total_area_px = sum(r['mask_area_px'] for r in regions)
    total_area_pct = round(total_area_px / total_pixels * 100, 3)

    if total_area_px > 0:
        weighted_severity = sum(
            r['severity_score'] * r['mask_area_px'] for r in regions
        ) / total_area_px
    else:
        weighted_severity = 0.0

    _, overall_label = get_severity_band(weighted_severity)

    damage_by_class = {}
    for cls in CLASS_NAMES[1:]:
        cls_regions = [r for r in regions if r['class'] == cls]
        damage_by_class[cls] = {
            'count': len(cls_regions),
            'total_area_pct': round(
                sum(r['mask_area_pct'] for r in cls_regions), 3
            ),
            'mean_severity': round(
                np.mean([r['severity_score'] for r in cls_regions]), 4
            ) if cls_regions else 0.0
        }

    return {
        'vehicle_overall_severity': round(float(weighted_severity), 4),
        'vehicle_overall_label': overall_label,
        'total_damage_area_pct': total_area_pct,
        'damage_count': len(regions),
        'damage_by_class': damage_by_class,
        'cost_estimate_usd': {
            'low': cost_lo_total,
            'high': cost_hi_total,
            'midpoint': (cost_lo_total + cost_hi_total) // 2
        },
        'regions': regions,
        'image_dimensions': {'height': orig_h, 'width': orig_w}
    }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def visualize_predictions(image: np.ndarray,
                           boxes: np.ndarray,
                           scores: np.ndarray,
                           classes: np.ndarray,
                           masks: List[np.ndarray],
                           severities: np.ndarray,
                           alpha: float = 0.4) -> np.ndarray:
    """
    Draw detection results on image.

    Args:
        image:      BGR image [H, W, 3]
        alpha:      mask overlay transparency
    Returns:
        annotated BGR image
    """
    vis = image.copy()
    overlay = image.copy()

    for box, score, cls_idx, mask, sev in zip(
        boxes, scores, classes, masks, severities
    ):
        cls_idx = int(cls_idx)
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) \
            else f'class_{cls_idx}'
        if cls_name == 'background':
            continue

        color = COLORS.get(cls_name, (200, 200, 200))
        _, sev_label = get_severity_band(float(sev))

        # Mask overlay
        mask_bool = mask > 0
        if mask_bool.any():
            overlay[mask_bool] = np.clip(
                0.5 * np.array(color) + 0.5 * overlay[mask_bool],
                0, 255
            ).astype(np.uint8)
            # Mask contour
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, color, 2)

        # Bounding box
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{cls_name} {score:.2f} | {sev_label} S:{sev:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    # Blend overlay
    result = cv2.addWeighted(vis, 1.0 - alpha, overlay, alpha, 0)

    # Add overall damage info
    _, overall_label = get_severity_band(
        float(np.mean(severities)) if len(severities) > 0 else 0.0
    )
    info = f"Detections: {len(boxes)} | Overall: {overall_label}"
    cv2.putText(result, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return result
