"""
Dataset Pipeline for Vehicle Damage Detection

Handles:
  - COCO-format JSON annotation loading
  - TFRecord writing and reading
  - tf.data pipeline with augmentation
  - Anchor target assignment
  - Batch collation
"""

import os
import json
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from src.data.preprocessing import (
    VehicleImagePreprocessor, build_augmentation_pipeline,
    letterbox_resize, normalize, IMAGENET_MEAN, IMAGENET_STD
)


# ─────────────────────────────────────────────────────────────────────────────
# COCO ANNOTATION LOADER
# ─────────────────────────────────────────────────────────────────────────────

class COCOAnnotationLoader:
    """
    Loads COCO-format annotations for vehicle damage dataset.

    Expected annotation structure:
    {
      "images": [{"id": 1, "file_name": "car.jpg", "width": 1920, "height": 1080}],
      "categories": [{"id": 1, "name": "scratch"}, ...],
      "annotations": [{
        "id": 1, "image_id": 1, "category_id": 1,
        "bbox": [x, y, w, h],  # COCO format (top-left + w/h)
        "segmentation": [[x1,y1,...,xn,yn]],
        "area": 1240.5,
        "severity": 0.35,
        "occlusion_level": 0.1,
        "reflection_affected": false
      }]
    }
    """

    CLASS_NAMES = ['background', 'scratch', 'dent',
                   'glass_damage', 'tear', 'body_deform']

    def __init__(self, annotation_path: str, image_dir: str):
        self.image_dir = Path(image_dir)

        with open(annotation_path) as f:
            self.coco = json.load(f)

        # Build lookup maps
        self.images_map = {
            img['id']: img for img in self.coco['images']
        }
        self.cat_to_label = {
            cat['id']: self.CLASS_NAMES.index(cat['name'])
            for cat in self.coco['categories']
            if cat['name'] in self.CLASS_NAMES
        }

        # Group annotations by image
        self.img_to_anns: Dict[int, List] = {}
        for ann in self.coco['annotations']:
            iid = ann['image_id']
            if iid not in self.img_to_anns:
                self.img_to_anns[iid] = []
            self.img_to_anns[iid].append(ann)

    def __len__(self):
        return len(self.coco['images'])

    def get_sample(self, idx: int) -> Dict:
        """Load image and annotations for given index."""
        img_info = self.coco['images'][idx]
        img_id = img_info['id']
        img_path = self.image_dir / img_info['file_name']

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        anns = self.img_to_anns.get(img_id, [])

        boxes, labels, severities, occlusions, masks = [], [], [], [], []
        H, W = img_info['height'], img_info['width']

        for ann in anns:
            if ann.get('iscrowd', 0):
                continue

            # COCO bbox: [x, y, w, h] → [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            x2, y2 = x + w, y + h
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x2), min(H, y2)

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            cat_id = ann['category_id']
            label = self.cat_to_label.get(cat_id, 0)

            boxes.append([x1, y1, x2, y2])
            labels.append(label)
            severities.append(float(ann.get('severity', 0.5)))
            occlusions.append(float(ann.get('occlusion_level', 0.0)))

            # Rasterize polygon segmentation to binary mask
            mask = self._rasterize_mask(ann.get('segmentation', []), H, W)
            masks.append(mask)

        return {
            'image': image,
            'boxes': np.array(boxes, dtype=np.float32).reshape(-1, 4),
            'labels': np.array(labels, dtype=np.int32),
            'severities': np.array(severities, dtype=np.float32),
            'occlusions': np.array(occlusions, dtype=np.float32),
            'masks': np.stack(masks, axis=0) if masks else
                     np.zeros((0, H, W), dtype=np.uint8),
            'image_id': img_id,
            'image_path': str(img_path)
        }

    def _rasterize_mask(self, segmentation: List, H: int, W: int) -> np.ndarray:
        """Convert polygon segmentation to binary mask."""
        mask = np.zeros((H, W), dtype=np.uint8)
        for seg in segmentation:
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# TFRECORD WRITER
# ─────────────────────────────────────────────────────────────────────────────

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tfrecords(loader: COCOAnnotationLoader,
                    output_path: str,
                    target_size: Tuple[int, int] = (640, 640),
                    shard_size: int = 500):
    """
    Convert COCO dataset to TFRecord format for efficient tf.data loading.

    Args:
        loader:      COCOAnnotationLoader instance
        output_path: path to write .tfrecord file(s)
        target_size: image resize target
        shard_size:  max records per shard file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    preprocessor = VehicleImagePreprocessor(target_size)
    shard_idx = 0
    writer = None
    records_written = 0

    for idx in tqdm(range(len(loader)), desc=f"Writing TFRecords"):
        # Start new shard
        if records_written % shard_size == 0:
            if writer:
                writer.close()
            shard_path = output_path.replace(
                '.tfrecord', f'_{shard_idx:04d}.tfrecord'
            )
            writer = tf.io.TFRecordWriter(shard_path)
            shard_idx += 1

        try:
            sample = loader.get_sample(idx)
        except Exception as e:
            print(f"Warning: skipping sample {idx}: {e}")
            continue

        image = sample['image']
        orig_h, orig_w = image.shape[:2]

        # Preprocess
        proc_image, meta = preprocessor(image)

        # Scale boxes to letterboxed coordinates
        boxes = sample['boxes'].copy()
        if len(boxes) > 0:
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] * meta['scale']
                                 + meta['pad_left'])
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] * meta['scale']
                                 + meta['pad_top'])

        # Resize and flatten masks
        mask_h, mask_w = target_size[0] // 4, target_size[1] // 4
        resized_masks = []
        for m in sample['masks']:
            m_resized = cv2.resize(m, (mask_w, mask_h),
                                    interpolation=cv2.INTER_NEAREST)
            resized_masks.append(m_resized.flatten())

        # Serialize
        feature = {
            'image': _bytes_feature(
                proc_image.astype(np.float32).tobytes()
            ),
            'image_h': _int64_list_feature([target_size[0]]),
            'image_w': _int64_list_feature([target_size[1]]),
            'orig_h': _int64_list_feature([orig_h]),
            'orig_w': _int64_list_feature([orig_w]),
            'boxes': _float_list_feature(boxes.flatten().tolist()),
            'labels': _int64_list_feature(sample['labels'].tolist()),
            'severities': _float_list_feature(
                sample['severities'].tolist()
            ),
            'occlusions': _float_list_feature(
                sample['occlusions'].tolist()
            ),
            'num_objects': _int64_list_feature([len(sample['labels'])]),
            'mask_h': _int64_list_feature([mask_h]),
            'mask_w': _int64_list_feature([mask_w]),
            'masks': _bytes_feature(
                np.stack(resized_masks, axis=0).astype(np.uint8).tobytes()
                if resized_masks else b''
            ),
            'image_id': _int64_list_feature([sample['image_id']]),
        }

        record = tf.train.Example(
            features=tf.train.Features(feature=feature)
        )
        writer.write(record.SerializeToString())
        records_written += 1

    if writer:
        writer.close()

    print(f"Wrote {records_written} records across {shard_idx} shards")


# ─────────────────────────────────────────────────────────────────────────────
# TFRECORD PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_tfrecord(serialized: tf.Tensor,
                   image_size: Tuple[int, int] = (640, 640),
                   num_classes: int = 6,
                   max_objects: int = 100) -> Tuple:
    """Parse a single TFRecord example."""
    mask_h, mask_w = image_size[0] // 4, image_size[1] // 4

    feature_desc = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_h': tf.io.FixedLenFeature([1], tf.int64),
        'image_w': tf.io.FixedLenFeature([1], tf.int64),
        'boxes': tf.io.VarLenFeature(tf.float32),
        'labels': tf.io.VarLenFeature(tf.int64),
        'severities': tf.io.VarLenFeature(tf.float32),
        'occlusions': tf.io.VarLenFeature(tf.float32),
        'num_objects': tf.io.FixedLenFeature([1], tf.int64),
        'masks': tf.io.FixedLenFeature([], tf.string),
        'mask_h': tf.io.FixedLenFeature([1], tf.int64),
        'mask_w': tf.io.FixedLenFeature([1], tf.int64),
        'image_id': tf.io.FixedLenFeature([1], tf.int64),
    }

    parsed = tf.io.parse_single_example(serialized, feature_desc)

    # Decode image
    image = tf.io.decode_raw(parsed['image'], tf.float32)
    image = tf.reshape(image, [image_size[0], image_size[1], 3])

    # Decode boxes and labels
    num_obj = tf.cast(parsed['num_objects'][0], tf.int32)
    boxes = tf.sparse.to_dense(parsed['boxes'])
    boxes = tf.reshape(boxes, [-1, 4])
    labels = tf.cast(tf.sparse.to_dense(parsed['labels']), tf.int32)
    severities = tf.sparse.to_dense(parsed['severities'])
    occlusions = tf.sparse.to_dense(parsed['occlusions'])

    # Pad to max_objects
    pad_n = max_objects - tf.shape(boxes)[0]
    boxes = tf.pad(boxes, [[0, pad_n], [0, 0]])
    labels = tf.pad(labels, [[0, pad_n]])
    severities = tf.pad(severities, [[0, pad_n]])
    occlusions = tf.pad(occlusions, [[0, pad_n]])

    boxes = boxes[:max_objects]
    labels = labels[:max_objects]
    severities = severities[:max_objects]
    occlusions = occlusions[:max_objects]

    # Aggregate severity and occlusion (mean over valid objects)
    valid_mask = tf.cast(
        tf.sequence_mask(num_obj, max_objects), tf.float32
    )
    img_severity = tf.reduce_sum(severities * valid_mask) / \
                   tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    img_occlusion = tf.reduce_sum(occlusions * valid_mask) / \
                    tf.maximum(tf.reduce_sum(valid_mask), 1.0)

    targets = {
        'boxes': boxes,
        'labels': labels,
        'severities': severities,
        'severity': tf.reshape(img_severity, [1]),
        'area_fraction': tf.reshape(
            tf.cast(num_obj, tf.float32) / 100.0, [1]
        ),
        'occlusion': tf.reshape(img_occlusion, [1]),
        'num_objects': num_obj,
        'image_id': parsed['image_id'][0],
    }

    return image, targets


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(tfrecord_pattern: str,
                  cfg: dict,
                  split: str = 'train') -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from TFRecord shards.

    Args:
        tfrecord_pattern: glob pattern for tfrecord files
        cfg:              full config dict
        split:            'train' | 'val' | 'test'

    Returns:
        Batched, prefetched tf.data.Dataset
    """
    import glob
    files = sorted(glob.glob(tfrecord_pattern.replace('.tfrecord',
                                                       '_*.tfrecord')))
    if not files:
        # Fallback: try exact file
        files = [tfrecord_pattern]

    image_size = tuple(cfg['model']['image_size'])
    num_classes = cfg['model']['num_classes']
    batch_size = cfg['training']['batch_size']

    ds = tf.data.TFRecordDataset(
        files, num_parallel_reads=tf.data.AUTOTUNE
    )

    if split == 'train':
        ds = ds.shuffle(buffer_size=2000, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda x: parse_tfrecord(x, image_size, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(batch_size, drop_remainder=(split == 'train'))
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ─────────────────────────────────────────────────────────────────────────────
# DATASET STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_dataset_stats(loader: COCOAnnotationLoader) -> Dict:
    """Compute class distribution and severity statistics for the dataset."""
    class_counts = {name: 0 for name in loader.CLASS_NAMES}
    severity_values = []
    occlusion_values = []
    area_values = []

    for idx in tqdm(range(len(loader)), desc="Computing stats"):
        try:
            sample = loader.get_sample(idx)
            for label in sample['labels']:
                class_counts[loader.CLASS_NAMES[label]] += 1
            severity_values.extend(sample['severities'].tolist())
            occlusion_values.extend(sample['occlusions'].tolist())
        except Exception:
            continue

    stats = {
        'class_counts': class_counts,
        'total_instances': sum(class_counts.values()),
        'severity_mean': float(np.mean(severity_values)) if severity_values else 0,
        'severity_std': float(np.std(severity_values)) if severity_values else 0,
        'severity_hist': np.histogram(severity_values, bins=10,
                                       range=(0, 1))[0].tolist()
                         if severity_values else [],
        'occlusion_mean': float(np.mean(occlusion_values)) if occlusion_values else 0,
    }
    return stats
