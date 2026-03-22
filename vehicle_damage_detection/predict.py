"""
Inference Script for Vehicle Damage Detection

Usage:
    # Single image
    python predict.py --image car.jpg --model checkpoints/best_model

    # Directory of images
    python predict.py --input-dir images/ --model checkpoints/best_model --output-dir results/

    # With custom threshold
    python predict.py --image car.jpg --model checkpoints/best_model --threshold 0.5
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))
from src.model import build_model
from src.data.preprocessing import VehicleImagePreprocessor, inverse_letterbox
from src.utils.postprocessing import (
    non_maximum_suppression, extract_instance_masks,
    build_damage_report, visualize_predictions
)

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class VehicleDamageInferenceEngine:
    """
    Production inference engine for vehicle damage detection.

    Handles:
      - Model loading and warm-up
      - Image preprocessing
      - Batched inference
      - Post-processing (NMS, mask extraction)
      - Report generation
    """

    CLASS_NAMES = ['background', 'scratch', 'dent',
                   'glass_damage', 'tear', 'body_deform']

    def __init__(self, model_path: str, config_path: str = 'configs/config.yaml',
                 score_threshold: float = 0.45,
                 nms_iou_threshold: float = 0.5,
                 max_detections: int = 50):
        """
        Args:
            model_path:        path to saved model weights
            config_path:       path to config YAML
            score_threshold:   minimum detection confidence
            nms_iou_threshold: NMS IoU threshold
            max_detections:    max detections per image
        """
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        self.image_size = tuple(self.cfg['model']['image_size'])

        # Load model
        print(f"Loading model from: {model_path}")
        self.model = build_model(self.cfg)
        self.model.load_weights(model_path)

        # Warm up
        dummy = tf.zeros([1, *self.image_size, 3])
        _ = self.model(dummy, training=False)
        print("Model loaded and warmed up.")

        # Preprocessor
        self.preprocessor = VehicleImagePreprocessor(
            target_size=self.image_size,
            apply_clahe=True,
            apply_highlight_suppression=True
        )

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Run full inference pipeline on a single BGR image.

        Args:
            image: BGR uint8 numpy array [H, W, 3]
        Returns:
            annotated_image: BGR image with detections drawn
            damage_report:   structured damage assessment dict
        """
        orig_h, orig_w = image.shape[:2]

        # Preprocess
        proc_image, meta = self.preprocessor(image)
        image_tensor = tf.expand_dims(
            tf.constant(proc_image, dtype=tf.float32), 0
        )

        # Inference
        predictions = self.model(image_tensor, training=False)

        # Extract raw outputs
        cls_outputs = tf.concat(
            [tf.cast(c, tf.float32) for c in predictions['cls']], axis=1
        )  # [1, N, num_classes]
        box_outputs = tf.concat(
            [tf.cast(b, tf.float32) for b in predictions['box']], axis=1
        )  # [1, N, 4]
        mask_output = predictions['masks'][0].numpy()  # [H/4, W/4, C]
        severity_output = predictions['severity'][0].numpy()  # [1]
        occlusion_output = predictions['occlusion'][0].numpy()  # [1]

        cls_np = cls_outputs[0].numpy()  # [N, C]
        box_np = box_outputs[0].numpy()  # [N, 4]

        # Get class with highest score per anchor (exclude background)
        cls_scores = cls_np[:, 1:]  # skip background
        per_anchor_score = cls_scores.max(axis=-1)
        per_anchor_class = cls_scores.argmax(axis=-1) + 1  # offset back

        # Dummy box decode (use boxes directly as coordinates for now)
        # In production: apply decode_boxes() with AnchorGenerator
        boxes = box_np * np.array([orig_w, orig_h, orig_w, orig_h])

        # NMS
        filtered_boxes, filtered_scores, filtered_classes = \
            non_maximum_suppression(
                boxes, per_anchor_score, per_anchor_class,
                iou_threshold=self.nms_iou_threshold,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections
            )

        # Extract per-instance masks
        masks = extract_instance_masks(
            mask_output, filtered_boxes, filtered_classes,
            orig_h, orig_w
        )

        # Per-detection severity (use global severity + noise for demo)
        base_severity = float(severity_output[0])
        severities = np.clip(
            base_severity + np.random.normal(0, 0.1, len(filtered_boxes)),
            0.0, 1.0
        )
        occlusions = np.full(len(filtered_boxes), float(occlusion_output[0]))

        # Build report
        report = build_damage_report(
            filtered_boxes, filtered_scores, filtered_classes,
            masks, severities, occlusions, orig_h, orig_w
        )

        # Visualize
        annotated = visualize_predictions(
            image.copy(), filtered_boxes, filtered_scores,
            filtered_classes, masks, severities
        )

        return annotated, report

    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple]:
        """Process a batch of images."""
        return [self.predict(img) for img in images]

    def predict_from_path(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Load image from path and predict."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load: {image_path}")
        return self.predict(image)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH DIRECTORY PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_directory(engine: VehicleDamageInferenceEngine,
                       input_dir: str,
                       output_dir: str,
                       extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')):
    """
    Process all images in a directory.

    Args:
        engine:     inference engine
        input_dir:  directory with input images
        output_dir: directory to save results
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in extensions
    ]
    print(f"Found {len(image_files)} images in {input_dir}")

    all_reports = {}

    for img_file in image_files:
        print(f"Processing: {img_file.name}")
        try:
            annotated, report = engine.predict_from_path(str(img_file))

            # Save annotated image
            out_img = output_path / f"{img_file.stem}_annotated{img_file.suffix}"
            cv2.imwrite(str(out_img), annotated)

            # Save individual report
            out_json = output_path / f"{img_file.stem}_report.json"
            with open(out_json, 'w') as f:
                json.dump(report, f, indent=2)

            all_reports[img_file.name] = report

            print(f"  → Detections: {report['damage_count']} | "
                  f"Overall: {report['vehicle_overall_label']} | "
                  f"Cost: ${report['cost_estimate_usd']['low']}"
                  f"–${report['cost_estimate_usd']['high']}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    # Save consolidated report
    consolidated_path = output_path / 'batch_report.json'
    with open(consolidated_path, 'w') as f:
        json.dump(all_reports, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Consolidated report: {consolidated_path}")

    return all_reports


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Vehicle Damage Detection Inference"
    )
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Directory of images to process')
    parser.add_argument('--model', type=str,
                        default='checkpoints/best_model',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                        default='configs/config.yaml')
    parser.add_argument('--output-dir', type=str,
                        default='results/')
    parser.add_argument('--threshold', type=float, default=0.45,
                        help='Detection confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                        help='NMS IoU threshold')
    parser.add_argument('--max-detections', type=int, default=50)
    parser.add_argument('--show', action='store_true',
                        help='Display result (requires display)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    engine = VehicleDamageInferenceEngine(
        model_path=args.model,
        config_path=args.config,
        score_threshold=args.threshold,
        nms_iou_threshold=args.nms_threshold,
        max_detections=args.max_detections
    )

    if args.image:
        # Single image mode
        annotated, report = engine.predict_from_path(args.image)

        print("\n" + "=" * 50)
        print(" DAMAGE ASSESSMENT REPORT")
        print("=" * 50)
        print(json.dumps(report, indent=2))

        # Save
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        img_name = Path(args.image).stem
        cv2.imwrite(str(out_dir / f"{img_name}_annotated.jpg"), annotated)
        with open(out_dir / f"{img_name}_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        if args.show:
            cv2.imshow('Damage Detection', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif args.input_dir:
        # Batch mode
        process_directory(engine, args.input_dir, args.output_dir)

    else:
        print("Please provide --image or --input-dir")
