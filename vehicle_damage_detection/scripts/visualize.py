"""
Dataset Visualization & Analysis Tools

Provides utilities for:
  - Visualizing annotated training samples
  - Plotting class distribution and severity histograms
  - Confusion matrix generation
  - Per-class PR curves
  - Prediction error analysis

Usage:
    python scripts/visualize.py --mode samples --annotation-file data/annotations/train.json
    python scripts/visualize.py --mode stats --stats-file data/tfrecords/dataset_stats.json
    python scripts/visualize.py --mode predictions --predictions results/test_predictions.json
"""

import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.postprocessing import COLORS, CLASS_NAMES, SEVERITY_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def visualize_annotation_samples(annotation_file: str,
                                   image_dir: str,
                                   n_samples: int = 9,
                                   output_path: str = 'sample_grid.png'):
    """
    Display a grid of annotated training samples.
    Shows bounding boxes, polygon masks, severity scores.
    """
    from src.data.dataset import COCOAnnotationLoader

    loader = COCOAnnotationLoader(annotation_file, image_dir)
    indices = np.random.choice(len(loader), min(n_samples, len(loader)),
                                replace=False)

    grid_size = int(np.ceil(np.sqrt(n_samples)))
    fig, axes = plt.subplots(grid_size, grid_size,
                              figsize=(5 * grid_size, 5 * grid_size))
    axes = axes.flatten()

    for ax_idx, sample_idx in enumerate(indices):
        sample = loader.get_sample(int(sample_idx))
        image = cv2.cvtColor(sample['image'], cv2.COLOR_BGR2RGB)

        # Draw annotations
        for i, (box, label, sev) in enumerate(
            zip(sample['boxes'], sample['labels'], sample['severities'])
        ):
            cls_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else 'unknown'
            color = [c / 255.0 for c in COLORS.get(cls_name, (200, 200, 200))]

            # Draw mask if available
            if i < len(sample['masks']) and sample['masks'][i].sum() > 0:
                mask = sample['masks'][i]
                colored_mask = np.zeros_like(image, dtype=np.float32)
                for c_idx, c_val in enumerate(color[:3]):
                    colored_mask[:, :, c_idx] = mask * c_val * 255
                image = np.clip(
                    image.astype(np.float32) * 0.7 + colored_mask * 0.3,
                    0, 255
                ).astype(np.uint8)

            # Draw box
            x1, y1, x2, y2 = [int(v) for v in box]
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            axes[ax_idx].add_patch(rect)
            axes[ax_idx].text(
                x1, y1 - 5, f"{cls_name} S:{sev:.2f}",
                fontsize=7, color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5)
            )

        axes[ax_idx].imshow(image)
        axes[ax_idx].axis('off')
        axes[ax_idx].set_title(f"Sample {sample_idx}", fontsize=9)

    # Hide unused axes
    for ax in axes[len(indices):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Sample grid saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET STATISTICS PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset_stats(stats_file: str,
                        output_dir: str = 'plots/'):
    """
    Generate class distribution and severity histogram plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(stats_file) as f:
        stats = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Vehicle Damage Dataset Statistics', fontsize=16, fontweight='bold')

    # 1. Class distribution per split
    ax = axes[0, 0]
    splits = list(stats.keys())
    class_names = [c for c in CLASS_NAMES[1:] if c != 'background']
    x = np.arange(len(class_names))
    width = 0.8 / len(splits)
    colors_bar = ['#2196F3', '#4CAF50', '#FF9800']

    for i, split in enumerate(splits):
        counts = [stats[split]['class_counts'].get(c, 0) for c in class_names]
        ax.bar(x + i * width, counts, width, label=split, color=colors_bar[i])

    ax.set_xlabel('Damage Class')
    ax.set_ylabel('Instance Count')
    ax.set_title('Class Distribution per Split')
    ax.set_xticks(x + width * len(splits) / 2)
    ax.set_xticklabels(class_names, rotation=20, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Severity histogram
    ax = axes[0, 1]
    for i, split in enumerate(splits):
        hist = stats[split].get('severity_hist', [])
        if hist:
            bins = np.linspace(0, 1, len(hist) + 1)
            ax.bar(bins[:-1], hist, width=0.1, alpha=0.6,
                    label=split, color=colors_bar[i])
    ax.axvline(0.30, color='orange', linestyle='--', label='Minor/Moderate')
    ax.axvline(0.65, color='red', linestyle='--', label='Moderate/Severe')
    ax.set_xlabel('Severity Score')
    ax.set_ylabel('Count')
    ax.set_title('Severity Score Distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Total instances per class (all splits combined)
    ax = axes[1, 0]
    total_counts = {}
    for split in splits:
        for cls, cnt in stats[split]['class_counts'].items():
            if cls != 'background':
                total_counts[cls] = total_counts.get(cls, 0) + cnt

    cls_sorted = sorted(total_counts, key=total_counts.get, reverse=True)
    bar_colors = ['#' + ''.join(f'{c:02x}' for c in COLORS.get(cls, (150, 150, 150)))
                  for cls in cls_sorted]
    ax.barh(cls_sorted, [total_counts[c] for c in cls_sorted],
             color=bar_colors)
    ax.set_xlabel('Total Instances')
    ax.set_title('Total Instances per Class')
    ax.grid(axis='x', alpha=0.3)

    # 4. Occlusion mean per split
    ax = axes[1, 1]
    occ_means = [stats[s].get('occlusion_mean', 0) for s in splits]
    sev_means = [stats[s].get('severity_mean', 0) for s in splits]
    width = 0.35
    x = np.arange(len(splits))
    ax.bar(x - width / 2, occ_means, width, label='Mean Occlusion', color='#9C27B0')
    ax.bar(x + width / 2, sev_means, width, label='Mean Severity', color='#F44336')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel('Score [0, 1]')
    ax.set_title('Average Occlusion & Severity per Split')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_file = output_path / 'dataset_stats.png'
    plt.savefig(str(out_file), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Statistics plot saved: {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_evaluation_results(metrics_file: str,
                              output_dir: str = 'plots/'):
    """Plot per-class AP, confusion matrix, severity scatter."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(metrics_file) as f:
        metrics = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Evaluation Results', fontsize=15, fontweight='bold')

    # 1. Per-class AP@50
    ax = axes[0]
    per_class_ap = metrics.get('per_class_ap50', {})
    if per_class_ap:
        classes = list(per_class_ap.keys())
        aps = list(per_class_ap.values())
        bar_colors = ['#' + ''.join(f'{c:02x}' for c in
                                     COLORS.get(cls, (150, 150, 150)))
                      for cls in classes]
        bars = ax.bar(classes, aps, color=bar_colors)
        ax.axhline(metrics.get('map_50', 0), color='black',
                   linestyle='--', label=f"mAP@50={metrics.get('map_50', 0):.3f}")
        for bar, ap in zip(bars, aps):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f'{ap:.3f}',
                    ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('AP@50')
        ax.set_title('Per-Class Average Precision')
        ax.set_xticklabels(classes, rotation=20, ha='right')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    # 2. Metric summary bar chart
    ax = axes[1]
    summary_metrics = {
        'mAP@50': metrics.get('map_50', 0),
        'mAP@50:95': metrics.get('map_5095', 0),
        'mIoU': metrics.get('miou', 0),
        'Pixel Acc': metrics.get('pixel_accuracy', 0),
        'Sev Band Acc': metrics.get('severity_band_acc', 0),
        'F1 Score': metrics.get('mean_f1', 0),
    }
    colors_summary = ['#2196F3', '#03A9F4', '#4CAF50', '#8BC34A',
                       '#FF9800', '#9C27B0']
    bars = ax.bar(summary_metrics.keys(), summary_metrics.values(),
                   color=colors_summary)
    for bar, val in zip(bars, summary_metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Summary')
    ax.set_xticklabels(summary_metrics.keys(), rotation=20, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # 3. Per-band severity breakdown
    ax = axes[2]
    per_band = metrics.get('severity_per_band', {})
    if per_band:
        bands = list(per_band.keys())
        maes = [per_band[b]['mae'] for b in bands]
        counts = [per_band[b]['count'] for b in bands]

        ax2 = ax.twinx()
        ax.bar(bands, maes, color=['#4CAF50', '#FF9800', '#F44336'],
                alpha=0.8)
        ax2.plot(bands, counts, 'ko--', label='Count')

        ax.set_ylabel('MAE', color='#333333')
        ax2.set_ylabel('Sample Count', color='black')
        ax.set_title('Severity MAE per Band')
        ax.set_ylim(0, max(maes) * 1.5 if maes else 0.3)
        for i, (band, mae, cnt) in enumerate(zip(bands, maes, counts)):
            ax.text(i, mae + 0.005, f'{mae:.3f}', ha='center', fontsize=9)
        ax2.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_file = output_path / 'evaluation_results.png'
    plt.savefig(str(out_file), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evaluation plot saved: {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CURVE PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(log_dir: str, output_dir: str = 'plots/'):
    """
    Parse TensorBoard event files and plot training/validation curves.
    Requires: pip install tensorboard
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("tensorboard required: pip install tensorboard")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    if not tags:
        print(f"No scalar data found in: {log_dir}")
        return

    # Group train/val
    train_tags = [t for t in tags if 'train' in t]
    val_tags = [t for t in tags if 'val' in t]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Curves', fontsize=15, fontweight='bold')

    pairs = [
        ('train/total_loss', 'val/train_loss', 'Total Loss', axes[0, 0]),
        ('val/map_50', None, 'mAP@50', axes[0, 1]),
        ('val/miou', None, 'mIoU', axes[0, 2]),
        ('val/severity_mae', None, 'Severity MAE', axes[1, 0]),
        ('train/cls_loss', None, 'Classification Loss', axes[1, 1]),
        ('train/severity_loss', None, 'Severity Loss', axes[1, 2]),
    ]

    for train_tag, val_tag, title, ax in pairs:
        for tag, color, label in [
            (train_tag, '#2196F3', 'train'),
            (val_tag, '#F44336', 'val')
        ]:
            if tag and tag in ea.Tags().get('scalars', []):
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                ax.plot(steps, values, color=color, label=label, linewidth=1.5)

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_file = output_path / 'training_curves.png'
    plt.savefig(str(out_file), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualization tools for vehicle damage detection"
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['samples', 'stats', 'evaluation', 'training'])
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--image-dir', type=str)
    parser.add_argument('--stats-file', type=str)
    parser.add_argument('--metrics-file', type=str)
    parser.add_argument('--log-dir', type=str)
    parser.add_argument('--output-dir', type=str, default='plots/')
    parser.add_argument('--n-samples', type=int, default=9)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'samples':
        assert args.annotation_file and args.image_dir, \
            "Samples mode requires --annotation-file and --image-dir"
        visualize_annotation_samples(
            args.annotation_file, args.image_dir,
            n_samples=args.n_samples,
            output_path=str(Path(args.output_dir) / 'sample_grid.png')
        )

    elif args.mode == 'stats':
        assert args.stats_file, "Stats mode requires --stats-file"
        plot_dataset_stats(args.stats_file, args.output_dir)

    elif args.mode == 'evaluation':
        assert args.metrics_file, "Evaluation mode requires --metrics-file"
        plot_evaluation_results(args.metrics_file, args.output_dir)

    elif args.mode == 'training':
        assert args.log_dir, "Training mode requires --log-dir"
        plot_training_curves(args.log_dir, args.output_dir)
