"""
Loss Functions for Vehicle Damage Detection

Includes:
  - Focal Loss (classification, handles class imbalance)
  - Smooth L1 Loss (box regression)
  - Dice Loss (segmentation)
  - Binary Cross-Entropy (mask)
  - Huber Loss (severity regression)
  - Combined multi-task loss
"""

import tensorflow as tf
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0,
               class_weights=None, from_logits=False):
    """
    Focal Loss for dense object detection (RetinaNet-style).

    FL(p) = -alpha * (1 - p)^gamma * log(p)

    Handles extreme foreground/background imbalance (~100:1).

    Args:
        y_true: [B, N, C] one-hot encoded target classes
        y_pred: [B, N, C] predicted probabilities (after sigmoid)
        alpha:  weighting factor for rare class
        gamma:  focusing parameter (0 = CE, 2 = standard focal)
        class_weights: optional per-class weight tensor [C]
    """
    if from_logits:
        y_pred = tf.sigmoid(y_pred)

    # Clamp for numerical stability
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    # Focal weights
    pt_pos = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_neg = tf.where(tf.equal(y_true, 0), 1 - y_pred, tf.ones_like(y_pred))

    focal_pos = alpha * tf.pow(1.0 - pt_pos, gamma) * (-tf.math.log(pt_pos))
    focal_neg = (1 - alpha) * tf.pow(pt_neg, gamma) * \
                (-tf.math.log(1.0 - tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)))

    loss = tf.where(tf.equal(y_true, 1), focal_pos, focal_neg)

    # Apply optional per-class weighting
    if class_weights is not None:
        loss = loss * tf.cast(class_weights, tf.float32)

    # Normalize by number of positive anchors
    num_pos = tf.maximum(tf.reduce_sum(tf.cast(y_true, tf.float32)), 1.0)
    return tf.reduce_sum(loss) / num_pos


# ─────────────────────────────────────────────────────────────────────────────
# SMOOTH L1 LOSS (HUBER)
# ─────────────────────────────────────────────────────────────────────────────

def smooth_l1_loss(y_true, y_pred, delta=1.0, weights=None):
    """
    Smooth L1 loss for bounding box regression.

    L(x) = 0.5*x^2       if |x| < delta
           delta*(|x| - 0.5*delta)  otherwise

    Args:
        y_true:  [B, N, 4] target box deltas (dx,dy,dw,dh)
        y_pred:  [B, N, 4] predicted box deltas
        delta:   transition point between L1 and L2
        weights: optional mask [B, N] for valid anchors
    """
    diff = tf.abs(y_true - y_pred)
    loss = tf.where(diff < delta,
                    0.5 * tf.square(diff),
                    delta * (diff - 0.5 * delta))
    loss = tf.reduce_sum(loss, axis=-1)  # sum over 4 coords

    if weights is not None:
        loss = loss * tf.cast(weights, tf.float32)

    num_pos = tf.maximum(
        tf.reduce_sum(tf.cast(weights if weights is not None
                              else tf.ones_like(loss), tf.float32)), 1.0
    )
    return tf.reduce_sum(loss) / num_pos


# ─────────────────────────────────────────────────────────────────────────────
# DICE LOSS
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss(y_true, y_pred, smooth=1.0):
    """
    Dice Loss for binary segmentation masks.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice

    Better than BCE alone for highly imbalanced mask pixels.

    Args:
        y_true: [B, H, W, C] binary ground truth masks
        y_pred: [B, H, W, C] predicted mask probabilities
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Flatten spatial dims
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=-1)
    dice = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_flat, axis=-1) +
        tf.reduce_sum(y_pred_flat, axis=-1) + smooth
    )
    return tf.reduce_mean(1.0 - dice)


# ─────────────────────────────────────────────────────────────────────────────
# BINARY CROSS-ENTROPY FOR MASKS
# ─────────────────────────────────────────────────────────────────────────────

def mask_bce_loss(y_true, y_pred):
    """
    Binary cross-entropy for mask prediction.
    Combined with dice for robust mask training.

    Args:
        y_true: [B, H, W, C] binary ground truth masks
        y_pred: [B, H, W, C] predicted probabilities
    """
    bce = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.MEAN
    )
    return bce(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED MASK LOSS
# ─────────────────────────────────────────────────────────────────────────────

def combined_mask_loss(y_true, y_pred, bce_weight=1.0, dice_weight=1.0):
    """Combines BCE + Dice for robust segmentation training."""
    bce = mask_bce_loss(y_true, y_pred)
    dc = dice_loss(y_true, y_pred)
    return bce_weight * bce + dice_weight * dc


# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY LOSS
# ─────────────────────────────────────────────────────────────────────────────

def severity_loss(y_true, y_pred, delta=0.5):
    """
    Huber loss for severity regression.
    More robust than MSE to annotation noise in severity labels.

    Args:
        y_true: [B, 1] ground truth severity scores [0,1]
        y_pred: [B, 1] predicted severity scores [0,1]
    """
    huber = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.MEAN)
    return huber(y_true, y_pred)


def area_fraction_loss(y_true, y_pred):
    """MSE for area fraction prediction."""
    return tf.reduce_mean(tf.square(y_true - y_pred))


def occlusion_loss(y_true, y_pred):
    """BCE for occlusion level estimation."""
    bce = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.MEAN
    )
    return bce(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TASK LOSS
# ─────────────────────────────────────────────────────────────────────────────

class VehicleDamageLoss:
    """
    Combined multi-task loss for vehicle damage detection.

    total = w_cls * focal_loss
          + w_box * smooth_l1
          + w_mask * (bce + dice)
          + w_severity * huber
          + w_occlusion * bce

    Usage:
        loss_fn = VehicleDamageLoss(cfg)
        total, loss_dict = loss_fn(predictions, targets)
    """

    def __init__(self, cfg: dict):
        weights = cfg['training']['loss_weights']
        self.w_cls = weights.get('cls', 1.0)
        self.w_box = weights.get('box', 1.0)
        self.w_mask = weights.get('mask', 2.0)
        self.w_dice = weights.get('dice', 0.5)
        self.w_severity = weights.get('severity', 1.5)
        self.w_occlusion = 0.5

        self.alpha = 0.25
        self.gamma = 2.0
        self.num_classes = cfg['model']['num_classes']

        # Class weights tensor (excluding background)
        cw = cfg['training'].get('class_weights', {})
        class_names = cfg['classes']
        self.class_weights = tf.constant(
            [cw.get(c, 1.0) for c in class_names], dtype=tf.float32
        )

    def __call__(self, predictions, targets):
        """
        Args:
            predictions: dict from model forward pass
            targets: dict with keys:
                'cls'      - [B, N, num_classes] one-hot
                'box'      - [B, N, 4] box deltas
                'box_mask' - [B, N] positive anchor mask
                'masks'    - [B, H, W, num_classes]
                'severity' - [B, 1]
                'area_fraction' - [B, 1]
                'occlusion' - [B, 1]
        Returns:
            total_loss (scalar), loss_dict (component breakdown)
        """
        loss_dict = {}

        # ── Classification loss ──────────────────────────────────────────
        pred_cls = tf.concat(predictions['cls'], axis=1)   # [B, all_A, C]
        tgt_cls = targets['cls']
        cls = focal_loss(tgt_cls, pred_cls,
                         alpha=self.alpha, gamma=self.gamma,
                         class_weights=self.class_weights)
        loss_dict['cls'] = cls

        # ── Box regression loss ──────────────────────────────────────────
        pred_box = tf.concat(predictions['box'], axis=1)   # [B, all_A, 4]
        tgt_box = targets['box']
        box_mask = targets.get('box_mask', None)
        box = smooth_l1_loss(tgt_box, pred_box, weights=box_mask)
        loss_dict['box'] = box

        # ── Mask loss (BCE + Dice) ───────────────────────────────────────
        pred_masks = predictions['masks']
        tgt_masks = targets['masks']
        mask_bce = mask_bce_loss(tgt_masks, pred_masks)
        mask_dice = dice_loss(tgt_masks, pred_masks)
        loss_dict['mask_bce'] = mask_bce
        loss_dict['mask_dice'] = mask_dice

        # ── Severity loss ────────────────────────────────────────────────
        sev = severity_loss(targets['severity'], predictions['severity'])
        loss_dict['severity'] = sev

        # ── Area fraction loss ───────────────────────────────────────────
        area = area_fraction_loss(targets['area_fraction'],
                                   predictions['area_fraction'])
        loss_dict['area_fraction'] = area

        # ── Occlusion loss ───────────────────────────────────────────────
        occ = occlusion_loss(targets['occlusion'], predictions['occlusion'])
        loss_dict['occlusion'] = occ

        # ── Total ────────────────────────────────────────────────────────
        total = (
            self.w_cls * cls +
            self.w_box * box +
            self.w_mask * mask_bce +
            self.w_dice * mask_dice +
            self.w_severity * sev +
            0.3 * area +
            self.w_occlusion * occ
        )
        loss_dict['total'] = total

        return total, loss_dict
