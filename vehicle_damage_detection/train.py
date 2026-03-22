"""
Training Script for Vehicle Damage Detection

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --resume checkpoints/ckpt-10
"""

import os
import sys
import argparse
import yaml
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ── Local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.model import build_model
from src.model.losses import VehicleDamageLoss
from src.data.dataset import build_dataset
from src.evaluation.metrics import VehicleDamageEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Vehicle Damage Detector")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--debug', action='store_true',
                        help='Run with small dataset for debugging')
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup followed by cosine decay.
    Stabilizes early training; improves final convergence.
    """

    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)

        # Warmup phase
        warmup_lr = self.base_lr * step / tf.maximum(warmup, 1.0)

        # Cosine decay phase
        progress = (step - warmup) / tf.maximum(total - warmup, 1.0)
        cosine_lr = self.base_lr * 0.5 * (
            1.0 + tf.cos(np.pi * tf.minimum(progress, 1.0))
        )

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            'base_lr': self.base_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps
        }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING STEP
# ─────────────────────────────────────────────────────────────────────────────

@tf.function
def train_step(model, optimizer, loss_fn, images, targets):
    """Single training step with gradient tape."""
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)

        # Build cls/box targets from GT boxes (simplified for clarity)
        # In production, use anchor assignment (AnchorTargetCreator)
        targets_for_loss = {
            'cls': tf.one_hot(targets['labels'], model.num_classes),
            'box': targets['boxes'],
            'box_mask': tf.cast(targets['labels'] > 0, tf.float32),
            'masks': tf.zeros_like(predictions['masks']),  # placeholder
            'severity': targets['severity'],
            'area_fraction': targets['area_fraction'],
            'occlusion': targets['occlusion'],
        }

        total, loss_dict = loss_fn(predictions, targets_for_loss)

    gradients = tape.gradient(total, model.trainable_variables)
    gradients, global_norm = tf.clip_by_global_norm(
        gradients, 10.0
    )
    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )
    return total, loss_dict, global_norm


@tf.function
def val_step(model, images):
    """Inference-only step for validation."""
    return model(images, training=False)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """Manages model checkpoints with best-model tracking."""

    def __init__(self, model, optimizer, ckpt_dir: str,
                 monitor: str = 'map_50', max_to_keep: int = 5):
        self.monitor = monitor
        self.best_value = 0.0
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        self.manager = tf.train.CheckpointManager(
            self.ckpt, str(self.ckpt_dir), max_to_keep=max_to_keep
        )

    def save(self, metrics: dict, epoch: int) -> bool:
        """Save checkpoint; returns True if this is the new best model."""
        current = metrics.get(self.monitor, 0.0)
        self.manager.save()

        is_best = current > self.best_value
        if is_best:
            self.best_value = current
            best_path = self.ckpt_dir / 'best_model'
            self.ckpt.write(str(best_path))

        return is_best

    def restore(self, path: Optional[str] = None):
        """Restore from checkpoint."""
        restore_path = path or self.manager.latest_checkpoint
        if restore_path:
            self.ckpt.restore(restore_path)
            print(f"Restored checkpoint from: {restore_path}")
        else:
            print("No checkpoint found, starting from scratch")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict, resume: Optional[str] = None,
          use_wandb: bool = True, debug: bool = False):
    """
    Full training loop.

    Args:
        cfg:       config dict from YAML
        resume:    checkpoint path to resume from
        use_wandb: enable W&B experiment tracking
        debug:     use tiny dataset for debugging
    """
    # ── Setup ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"vehicle_damage_{timestamp}"

    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg['project']['name'],
                name=run_name,
                config=cfg
            )
        except ImportError:
            print("wandb not installed, skipping")
            use_wandb = False

    # Mixed precision
    if cfg['training'].get('mixed_precision', True):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Using mixed precision (float16)")

    # ── GPU Setup ──────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # ── Datasets ───────────────────────────────────────────────────────────
    print("Building datasets...")
    train_ds = build_dataset(
        cfg['data']['train_tfrecord'], cfg, split='train'
    )
    val_ds = build_dataset(
        cfg['data']['val_tfrecord'], cfg, split='val'
    )

    if debug:
        train_ds = train_ds.take(20)
        val_ds = val_ds.take(10)
        print("DEBUG mode: using 20 train / 10 val batches")

    # ── Model ──────────────────────────────────────────────────────────────
    print("Building model...")
    model = build_model(cfg)
    print(f"Model parameters: "
          f"{sum(tf.size(v).numpy() for v in model.trainable_variables):,}")

    # ── Optimizer ──────────────────────────────────────────────────────────
    steps_per_epoch = 1000  # Approximate; update after first epoch
    total_steps = cfg['training']['epochs'] * steps_per_epoch
    warmup_steps = cfg['training']['warmup_epochs'] * steps_per_epoch

    lr_schedule = WarmupCosineDecay(
        base_lr=cfg['training']['learning_rate'],
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=cfg['training']['weight_decay']
    )

    # ── Loss ───────────────────────────────────────────────────────────────
    loss_fn = VehicleDamageLoss(cfg)

    # ── Checkpoint & Logging ───────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(
        model, optimizer,
        cfg['training']['checkpoint_dir'],
        monitor=cfg['training'].get('monitor_metric', 'map_50')
    )

    if resume:
        ckpt_mgr.restore(resume)

    log_dir = Path(cfg['training']['log_dir']) / run_name
    summary_writer = tf.summary.create_file_writer(str(log_dir))

    # ── Evaluator ──────────────────────────────────────────────────────────
    evaluator = VehicleDamageEvaluator(cfg['model']['num_classes'])

    # ── Training Loop ──────────────────────────────────────────────────────
    best_map = 0.0
    global_step = 0

    print(f"\nStarting training for {cfg['training']['epochs']} epochs")
    print("=" * 60)

    for epoch in range(1, cfg['training']['epochs'] + 1):
        epoch_start = time.time()
        train_losses = []
        loss_components = {}

        # ── Train Epoch ─────────────────────────────────────────────────
        model.trainable = True
        pbar = tqdm(enumerate(train_ds),
                    desc=f"Epoch {epoch}/{cfg['training']['epochs']}",
                    leave=False)

        for step, (images, targets) in pbar:
            total_loss, loss_dict, grad_norm = train_step(
                model, optimizer, loss_fn, images, targets
            )

            train_losses.append(float(total_loss))
            global_step += 1

            # Accumulate component losses
            for k, v in loss_dict.items():
                if k not in loss_components:
                    loss_components[k] = []
                loss_components[k].append(float(v))

            pbar.set_postfix({
                'loss': f"{float(total_loss):.4f}",
                'lr': f"{float(optimizer.learning_rate(global_step)):.2e}"
            })

            # TensorBoard logging
            if global_step % 50 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar('train/total_loss', total_loss,
                                      step=global_step)
                    tf.summary.scalar('train/grad_norm', grad_norm,
                                      step=global_step)
                    for k, v in loss_dict.items():
                        tf.summary.scalar(f'train/{k}_loss', v,
                                          step=global_step)

        mean_train_loss = np.mean(train_losses)
        epoch_time = time.time() - epoch_start

        # ── Validation ─────────────────────────────────────────────────
        evaluator.reset()
        model.trainable = False

        for images, targets in tqdm(val_ds, desc="  Validating", leave=False):
            preds = val_step(model, images)
            evaluator.update(preds, targets)

        val_results = evaluator.compute()
        val_results['train_loss'] = mean_train_loss
        val_results['epoch_time_s'] = epoch_time
        val_results['lr'] = float(optimizer.learning_rate(global_step))

        # ── Logging ────────────────────────────────────────────────────
        with summary_writer.as_default():
            for k, v in val_results.items():
                if isinstance(v, (int, float)):
                    tf.summary.scalar(f'val/{k}', v, step=epoch)

        if use_wandb:
            try:
                import wandb
                wandb.log(val_results, step=epoch)
            except Exception:
                pass

        # Console output
        is_best = ckpt_mgr.save(val_results, epoch)
        best_marker = " ★ NEW BEST" if is_best else ""
        if is_best:
            best_map = val_results.get('map_50', 0.0)

        print(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']} | "
            f"Loss: {mean_train_loss:.4f} | "
            f"mAP@50: {val_results.get('map_50', 0):.4f} | "
            f"mIoU: {val_results.get('miou', 0):.4f} | "
            f"SevMAE: {val_results.get('severity_mae', 0):.4f} | "
            f"{epoch_time:.1f}s{best_marker}"
        )

    print(f"\nTraining complete. Best mAP@50: {best_map:.4f}")
    print(f"Best model saved to: {ckpt_mgr.ckpt_dir}/best_model")

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    return model


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from typing import Optional
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(
        cfg=cfg,
        resume=args.resume,
        use_wandb=not args.no_wandb,
        debug=args.debug
    )
