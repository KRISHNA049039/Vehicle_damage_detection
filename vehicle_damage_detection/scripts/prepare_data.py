"""
Data Preparation Script

Converts COCO-format annotations to TFRecord format for training.

Usage:
    python scripts/prepare_data.py \
        --train-annotations data/annotations/train.json \
        --train-images data/images/train \
        --val-annotations data/annotations/val.json \
        --val-images data/images/val \
        --output-dir data/tfrecords \
        --image-size 640

Also computes and saves dataset statistics.
"""

import sys
import argparse
import json
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import (
    COCOAnnotationLoader, write_tfrecords, compute_dataset_stats
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare vehicle damage dataset TFRecords"
    )
    parser.add_argument('--train-annotations', type=str, required=True)
    parser.add_argument('--train-images', type=str, required=True)
    parser.add_argument('--val-annotations', type=str, required=True)
    parser.add_argument('--val-images', type=str, required=True)
    parser.add_argument('--test-annotations', type=str, default=None)
    parser.add_argument('--test-images', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='data/tfrecords')
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--shard-size', type=int, default=500,
                        help='Max records per TFRecord shard')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only compute stats, skip TFRecord writing')
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_size = (args.image_size, args.image_size)
    splits = [('train', args.train_annotations, args.train_images)]
    splits.append(('val', args.val_annotations, args.val_images))
    if args.test_annotations and args.test_images:
        splits.append(('test', args.test_annotations, args.test_images))

    all_stats = {}

    for split_name, ann_path, img_dir in splits:
        print(f"\n{'='*50}")
        print(f"Processing {split_name} split...")
        print(f"  Annotations: {ann_path}")
        print(f"  Images:      {img_dir}")

        loader = COCOAnnotationLoader(ann_path, img_dir)
        print(f"  Dataset size: {len(loader)} images")

        # Compute statistics
        print(f"  Computing dataset statistics...")
        stats = compute_dataset_stats(loader)
        all_stats[split_name] = stats

        print(f"  Class distribution:")
        for cls, count in stats['class_counts'].items():
            print(f"    {cls:<18}: {count:>6}")
        print(f"  Total instances: {stats['total_instances']}")
        print(f"  Severity: mean={stats['severity_mean']:.3f}, "
              f"std={stats['severity_std']:.3f}")

        if not args.stats_only:
            # Write TFRecords
            out_path = str(output_dir / f'{split_name}.tfrecord')
            print(f"  Writing TFRecords to: {out_path}")
            write_tfrecords(
                loader, out_path,
                target_size=target_size,
                shard_size=args.shard_size
            )
            print(f"  ✓ {split_name} TFRecords written")

    # Save statistics
    stats_path = output_dir / 'dataset_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nDataset statistics saved to: {stats_path}")
    print("\nData preparation complete!")


if __name__ == '__main__':
    main()
