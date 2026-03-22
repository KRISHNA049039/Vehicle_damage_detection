"""
Anchor Target Creator

Assigns ground-truth boxes to anchors for training.
For each image, determines which anchors are positive (foreground),
negative (background), or ignored, and computes regression targets.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict
from src.model import AnchorGenerator


class AnchorTargetCreator:
    """
    Assigns GT boxes to anchors for computing training targets.

    For each anchor:
      - IoU >= pos_iou_thresh  → positive (foreground)
      - IoU <  neg_iou_thresh  → negative (background)
      - In between             → ignored

    Also always assigns the best anchor to each GT box (ensures
    every GT box has at least one anchor assigned).
    """

    def __init__(self,
                 pos_iou_thresh: float = 0.5,
                 neg_iou_thresh: float = 0.4,
                 pos_fraction: float = 0.5,
                 n_sample: int = 256,
                 box_weight: Tuple[float, ...] = (10., 10., 5., 5.)):
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_fraction = pos_fraction
        self.n_sample = n_sample
        self.box_weight = box_weight

    def __call__(self,
                 anchors: np.ndarray,
                 gt_boxes: np.ndarray,
                 gt_labels: np.ndarray,
                 gt_severities: np.ndarray,
                 image_h: int,
                 image_w: int) -> Dict:
        """
        Compute anchor targets for one image.

        Args:
            anchors:       [N, 4] all anchors for this image
            gt_boxes:      [M, 4] ground truth boxes [x1,y1,x2,y2]
            gt_labels:     [M]   GT class labels
            gt_severities: [M]   GT severity scores [0,1]
            image_h/w:     original image dimensions

        Returns:
            dict with:
              'cls_targets'  [N, num_classes] one-hot
              'box_targets'  [N, 4] encoded deltas
              'pos_mask'     [N] bool — positive anchors
              'neg_mask'     [N] bool — negative anchors
              'ignore_mask'  [N] bool — ignored anchors
              'matched_gt'   [N] int  — which GT box each anchor matches (-1 = none)
        """
        N = len(anchors)
        M = len(gt_boxes)

        # Clip anchors to image bounds
        anchors_clipped = np.clip(
            anchors,
            [0, 0, 0, 0],
            [image_w, image_h, image_w, image_h]
        )

        # Remove invalid (zero-area) anchors
        aw = anchors_clipped[:, 2] - anchors_clipped[:, 0]
        ah = anchors_clipped[:, 3] - anchors_clipped[:, 1]
        valid = (aw > 1) & (ah > 1)

        # Default: all ignored
        labels = np.full(N, -1, dtype=np.int32)   # -1 = ignore
        matched_gt = np.full(N, -1, dtype=np.int32)

        if M == 0:
            # No GT boxes: all valid anchors are negative
            labels[valid] = 0
            return self._build_targets(
                anchors, labels, matched_gt,
                gt_boxes, gt_labels, gt_severities
            )

        # Compute IoU between anchors and GT boxes
        from src.evaluation.metrics import compute_iou_matrix
        iou = compute_iou_matrix(anchors_clipped, gt_boxes)  # [N, M]

        # Best GT for each anchor
        best_gt_iou = iou.max(axis=1)   # [N]
        best_gt_idx = iou.argmax(axis=1)  # [N]

        # Best anchor for each GT
        best_anchor_iou = iou.max(axis=0)  # [M]
        best_anchor_idx = iou.argmax(axis=0)  # [M]

        # Assign negatives first
        labels[valid & (best_gt_iou < self.neg_iou_thresh)] = 0

        # Assign positives: IoU > threshold
        labels[valid & (best_gt_iou >= self.pos_iou_thresh)] = 1

        # Assign best anchor for each GT (guarantee each GT is covered)
        for gt_idx, anchor_idx in enumerate(best_anchor_idx):
            if best_anchor_iou[gt_idx] > 0:
                labels[anchor_idx] = 1

        # Record which GT each positive anchor matches
        pos_mask = labels == 1
        matched_gt[pos_mask] = best_gt_idx[pos_mask]

        # Subsample: limit positive/negative ratio
        labels = self._subsample(labels)

        return self._build_targets(
            anchors, labels, matched_gt,
            gt_boxes, gt_labels, gt_severities
        )

    def _subsample(self, labels: np.ndarray) -> np.ndarray:
        """Randomly subsample pos/neg anchors to n_sample total."""
        n_pos = int(self.n_sample * self.pos_fraction)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]

        # Downsample positives if too many
        if len(pos_idx) > n_pos:
            disable = np.random.choice(pos_idx, len(pos_idx) - n_pos,
                                        replace=False)
            labels[disable] = -1

        # Downsample negatives to fill remainder
        n_neg = self.n_sample - (labels == 1).sum()
        if len(neg_idx) > n_neg:
            disable = np.random.choice(neg_idx, len(neg_idx) - n_neg,
                                        replace=False)
            labels[disable] = -1

        return labels

    def _build_targets(self, anchors, labels, matched_gt,
                        gt_boxes, gt_labels, gt_severities) -> Dict:
        """Encode matched GT information into training targets."""
        N = len(anchors)
        num_classes = int(gt_labels.max()) + 1 if len(gt_labels) > 0 else 6
        num_classes = max(num_classes, 6)

        pos_mask = labels == 1
        neg_mask = labels == 0
        ignore_mask = labels == -1

        # One-hot class targets
        cls_targets = np.zeros((N, num_classes), dtype=np.float32)
        cls_targets[neg_mask, 0] = 1.0  # background class
        if pos_mask.sum() > 0:
            matched_cls = gt_labels[matched_gt[pos_mask]]
            cls_targets[pos_mask] = 0
            for i, (anchor_idx, cls) in enumerate(
                zip(np.where(pos_mask)[0], matched_cls)
            ):
                if cls < num_classes:
                    cls_targets[anchor_idx, cls] = 1.0

        # Box regression targets (only for positives)
        box_targets = np.zeros((N, 4), dtype=np.float32)
        if pos_mask.sum() > 0:
            pos_anchors = anchors[pos_mask]
            pos_gt = gt_boxes[matched_gt[pos_mask]]
            box_targets[pos_mask] = self._encode_boxes(pos_anchors, pos_gt)

        # Severity targets
        sev_targets = np.zeros(N, dtype=np.float32)
        if pos_mask.sum() > 0 and len(gt_severities) > 0:
            sev_targets[pos_mask] = gt_severities[matched_gt[pos_mask]]

        return {
            'cls_targets': cls_targets,
            'box_targets': box_targets,
            'pos_mask': pos_mask,
            'neg_mask': neg_mask,
            'ignore_mask': ignore_mask,
            'matched_gt': matched_gt,
            'sev_targets': sev_targets
        }

    def _encode_boxes(self, anchors: np.ndarray,
                       gt_boxes: np.ndarray) -> np.ndarray:
        """
        Encode GT boxes as deltas relative to anchors.

        tx = (x_gt - x_a) / w_a * wx
        ty = (y_gt - y_a) / h_a * wy
        tw = log(w_gt / w_a) * ww
        th = log(h_gt / h_a) * wh
        """
        wx, wy, ww, wh = self.box_weight

        # Anchor centers and dims
        aw = anchors[:, 2] - anchors[:, 0]
        ah = anchors[:, 3] - anchors[:, 1]
        ax = anchors[:, 0] + 0.5 * aw
        ay = anchors[:, 1] + 0.5 * ah

        # GT centers and dims
        gw = gt_boxes[:, 2] - gt_boxes[:, 0]
        gh = gt_boxes[:, 3] - gt_boxes[:, 1]
        gx = gt_boxes[:, 0] + 0.5 * gw
        gy = gt_boxes[:, 1] + 0.5 * gh

        # Encode
        aw = np.maximum(aw, 1e-6)
        ah = np.maximum(ah, 1e-6)
        gw = np.maximum(gw, 1e-6)
        gh = np.maximum(gh, 1e-6)

        tx = (gx - ax) / aw * wx
        ty = (gy - ay) / ah * wy
        tw = np.log(gw / aw) * ww
        th = np.log(gh / ah) * wh

        return np.stack([tx, ty, tw, th], axis=-1)


class BatchAnchorTargetCreator:
    """
    Applies AnchorTargetCreator over a batch of images.
    Returns batched TF tensors ready for loss computation.
    """

    def __init__(self, cfg: dict):
        from src.model import AnchorGenerator
        model_cfg = cfg['model']
        self.anchor_gen = AnchorGenerator(
            sizes=model_cfg['anchor_sizes'],
            ratios=model_cfg['anchor_ratios'],
            scales=model_cfg.get('anchor_scales', [1.0, 1.26, 1.587]),
            strides=[8, 16, 32, 64, 128]
        )
        self.target_creator = AnchorTargetCreator(
            pos_iou_thresh=cfg['training']['iou_threshold_pos'],
            neg_iou_thresh=cfg['training']['iou_threshold_neg']
        )
        self.image_size = tuple(model_cfg['image_size'])

    def __call__(self, targets: Dict) -> Dict:
        """
        Args:
            targets: batch dict from tf.data pipeline
        Returns:
            dict with batched anchor targets
        """
        anchors = self.anchor_gen.generate_anchors(self.image_size).numpy()

        gt_boxes_batch = targets['boxes'].numpy()
        gt_labels_batch = targets['labels'].numpy()
        gt_severities_batch = targets['severities'].numpy()
        num_objects_batch = targets['num_objects'].numpy()

        batch_cls, batch_box, batch_pos = [], [], []

        for b in range(len(gt_boxes_batch)):
            n = int(num_objects_batch[b])
            gt_boxes = gt_boxes_batch[b, :n]
            gt_labels = gt_labels_batch[b, :n]
            gt_sev = gt_severities_batch[b, :n]

            result = self.target_creator(
                anchors, gt_boxes, gt_labels, gt_sev,
                self.image_size[0], self.image_size[1]
            )
            batch_cls.append(result['cls_targets'])
            batch_box.append(result['box_targets'])
            batch_pos.append(result['pos_mask'].astype(np.float32))

        return {
            'cls': tf.constant(np.stack(batch_cls)),
            'box': tf.constant(np.stack(batch_box)),
            'box_mask': tf.constant(np.stack(batch_pos)),
            'masks': targets.get('masks',
                                  tf.zeros([len(gt_boxes_batch), 1, 1, 6])),
            'severity': targets['severity'],
            'area_fraction': targets['area_fraction'],
            'occlusion': targets['occlusion'],
        }
