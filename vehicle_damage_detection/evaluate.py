"""
Evaluation Script for Vehicle Damage Detection

Usage:
    python evaluate.py --model checkpoints/best_model --split test
    python evaluate.py --model checkpoints/best_model --split val --save-predictions
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import yaml
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.model import build_model
from src.data.dataset import build_dataset
from src.evaluation.metrics import VehicleDamageEvaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--output-dir', type=str, default='evaluation_results')
    return parser.parse_args()


def evaluate(cfg, model_path, split='test',
             save_predictions=False, output_dir='evaluation_results'):
    """Run full evaluation on a dataset split."""

    print(f"\nEvaluating on {split} split...")

    # Load model
    model = build_model(cfg)
    model.load_weights(model_path)
    print(f"Model loaded from: {model_path}")

    # Dataset
    tfrecord_path = cfg['data'][f'{split}_tfrecord']
    ds = build_dataset(tfrecord_path, cfg, split=split)

    evaluator = VehicleDamageEvaluator(cfg['model']['num_classes'])
    all_predictions = []

    for images, targets in tqdm(ds, desc="Evaluating"):
        predictions = model(images, training=False)
        evaluator.update(predictions, targets)

        if save_predictions:
            for b in range(images.shape[0]):
                all_predictions.append({
                    'image_id': int(targets['image_id'][b].numpy()),
                    'severity_pred': float(predictions['severity'][b, 0].numpy()),
                    'severity_gt': float(targets['severity'][b, 0].numpy()),
                    'occlusion_pred': float(predictions['occlusion'][b, 0].numpy()),
                })

    # Compute and display
    results = evaluator.compute()
    evaluator.print_report(results)

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f'{split}_metrics.json', 'w') as f:
        # Convert numpy types to native Python
        def convert(o):
            if isinstance(o, (np.floating, np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.integer, np.int32, np.int64)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o

        json.dump(
            {k: convert(v) for k, v in results.items()},
            f, indent=2, default=convert
        )
    print(f"\nMetrics saved to: {out_dir}/{split}_metrics.json")

    if save_predictions and all_predictions:
        with open(out_dir / f'{split}_predictions.json', 'w') as f:
            json.dump(all_predictions, f, indent=2)
        print(f"Predictions saved to: {out_dir}/{split}_predictions.json")

    return results


if __name__ == '__main__':
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(
        cfg=cfg,
        model_path=args.model,
        split=args.split,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )
