"""
Preprocessing and Augmentation Pipeline for Vehicle Damage Detection

Handles:
  - Letterbox resize (aspect-ratio preserving)
  - CLAHE contrast enhancement (scratch visibility)
  - Specular highlight suppression (metallic vehicle surfaces)
  - Comprehensive albumentations augmentation
  - Normalization with ImageNet stats
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def letterbox_resize(image: np.ndarray,
                     target: Tuple[int, int] = (640, 640),
                     color: Tuple[int, int, int] = (114, 114, 114)
                     ) -> Tuple[np.ndarray, Dict]:
    """
    Resize image with aspect-ratio preservation via letterboxing.

    Returns:
        resized_image: padded image of target size
        meta: dict with scale, pad_top, pad_left for inverse transform
    """
    h, w = image.shape[:2]
    target_h, target_w = target

    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    image = cv2.copyMakeBorder(
        image, pad_top, pad_bot, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=color
    )

    return image, {
        'scale': scale,
        'pad_top': pad_top,
        'pad_left': pad_left,
        'orig_h': h,
        'orig_w': w
    }


def inverse_letterbox(boxes: np.ndarray, meta: Dict,
                       target: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Convert boxes from letterboxed coordinates back to original image coordinates.

    Args:
        boxes: [N, 4] in [x1, y1, x2, y2] format
        meta:  dict from letterbox_resize
    """
    boxes = boxes.copy().astype(np.float32)
    boxes[:, [0, 2]] -= meta['pad_left']
    boxes[:, [1, 3]] -= meta['pad_top']
    boxes /= meta['scale']

    # Clip to original image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, meta['orig_w'])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, meta['orig_h'])

    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# SPECULAR HIGHLIGHT SUPPRESSION
# ─────────────────────────────────────────────────────────────────────────────

def suppress_specular_highlights(image: np.ndarray,
                                  threshold: int = 240,
                                  inpaint_radius: int = 5) -> np.ndarray:
    """
    Suppresses specular reflections on metallic vehicle surfaces.

    Strategy:
      1. Detect overexposed regions (intensity > threshold)
      2. Inpaint using TELEA algorithm (texture-aware)
      3. Blend: mild highlights preserved, extreme ones suppressed

    Args:
        image:          BGR image [H, W, 3]
        threshold:      pixel intensity above which we consider specular
        inpaint_radius: radius for inpainting algorithm
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    highlight_mask = (gray > threshold).astype(np.uint8) * 255

    if highlight_mask.sum() == 0:
        return image  # No highlights to suppress

    # Dilate mask slightly to cover highlight halos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    highlight_mask = cv2.dilate(highlight_mask, kernel, iterations=1)

    # Inpaint overexposed regions
    inpainted = cv2.inpaint(
        image, highlight_mask, inpaint_radius, cv2.INPAINT_TELEA
    )

    # Soft blend: suppress extreme highlights, preserve mild ones
    alpha = np.clip(
        (gray.astype(np.float32) - 200) / 55.0, 0.0, 1.0
    )[..., np.newaxis]
    result = (1.0 - alpha) * image.astype(np.float32) + \
             alpha * inpainted.astype(np.float32)

    return result.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# CLAHE ENHANCEMENT
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(image: np.ndarray,
                clip_limit: float = 2.0,
                tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.

    Applied in LAB color space to only affect luminance (L channel),
    preserving color fidelity while boosting local contrast.
    Especially effective for detecting scratches on low-contrast panels.

    Args:
        image:      BGR image
        clip_limit: upper limit for contrast amplification
        tile_size:  local neighborhood size for histogram computation
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize image with ImageNet mean/std.
    Input:  [H, W, 3] uint8 BGR
    Output: [H, W, 3] float32 normalized
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image


def denormalize(image: np.ndarray) -> np.ndarray:
    """Inverse of normalize — for visualization."""
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class VehicleImagePreprocessor:
    """
    Full preprocessing pipeline for inference.

    Steps:
        1. Letterbox resize
        2. Specular highlight suppression
        3. CLAHE enhancement
        4. ImageNet normalization
    """

    def __init__(self, target_size: Tuple[int, int] = (640, 640),
                 apply_clahe: bool = True,
                 apply_highlight_suppression: bool = True):
        self.target_size = target_size
        self.do_clahe = apply_clahe
        self.do_highlight_suppression = apply_highlight_suppression

    def __call__(self, image: np.ndarray):
        """
        Args:
            image: BGR uint8 numpy array [H, W, 3]
        Returns:
            processed: float32 normalized [H, W, 3]
            meta: letterbox metadata for coordinate inverse transform
        """
        # Step 1: Letterbox resize
        image, meta = letterbox_resize(image, self.target_size)

        # Step 2: Highlight suppression (reflections on metallic surfaces)
        if self.do_highlight_suppression:
            image = suppress_specular_highlights(image)

        # Step 3: CLAHE (local contrast enhancement for scratch visibility)
        if self.do_clahe:
            image = apply_clahe(image)

        # Step 4: Normalize
        processed = normalize(image)

        return processed, meta

    def batch_preprocess(self, images: List[np.ndarray]):
        """Preprocess a batch of images."""
        processed = []
        metas = []
        for img in images:
            p, m = self(img)
            processed.append(p)
            metas.append(m)
        return np.stack(processed, axis=0), metas


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION PIPELINE (TRAINING)
# ─────────────────────────────────────────────────────────────────────────────

def build_augmentation_pipeline(cfg: dict, split: str = 'train') -> A.Compose:
    """
    Build albumentations augmentation pipeline from config.

    Training augmentations include:
      - Geometric: flip, rotate, scale, shift
      - Photometric: brightness, contrast, HSV, gamma
      - Reflection simulation: CLAHE, gamma
      - Occlusion simulation: CoarseDropout
      - Noise: Gaussian, ISO
      - Blur: Gaussian, motion

    Args:
        cfg:   full config dict
        split: 'train' | 'val' | 'test'

    Returns:
        albumentations Compose pipeline
    """
    aug = cfg.get('augmentation', {})

    bbox_params = A.BboxParams(
        format='coco',           # [x, y, width, height]
        label_fields=['category_ids', 'severity_labels'],
        min_visibility=0.3        # Remove boxes with < 30% visibility after crop
    )

    if split != 'train':
        # Val/test: only resize (handled separately) + normalize
        return A.Compose([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ], bbox_params=bbox_params)

    transforms = [
        # ── Geometric ────────────────────────────────────────────────────
        A.HorizontalFlip(p=aug.get('horizontal_flip', 0.5)),
        A.VerticalFlip(p=aug.get('vertical_flip', 0.0)),

        A.ShiftScaleRotate(
            shift_limit=aug.get('shift_limit', 0.05),
            scale_limit=aug.get('scale_limit', 0.1),
            rotate_limit=aug.get('rotation_limit', 15),
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
            p=0.4
        ),

        A.Perspective(scale=(0.02, 0.1), p=0.2),

        # ── Photometric ───────────────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=aug.get('brightness_limit', 0.3),
            contrast_limit=aug.get('contrast_limit', 0.3),
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=aug.get('hue_shift', 10),
            sat_shift_limit=aug.get('sat_shift', 30),
            val_shift_limit=aug.get('val_shift', 20),
            p=0.4
        ),

        # ── Reflection simulation ─────────────────────────────────────────
        A.RandomGamma(gamma_limit=(80, 120), p=aug.get('gamma', 0.3)),
        A.CLAHE(clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=aug.get('clahe', 0.3)),

        # ── Specular highlight simulation ─────────────────────────────────
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_lower=0.5,
            num_flare_circles_lower=1,
            num_flare_circles_upper=3,
            src_radius=50,
            p=0.1
        ),

        # ── Blur ──────────────────────────────────────────────────────────
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
            A.MedianBlur(blur_limit=5),
        ], p=aug.get('gaussian_blur', 0.2)),

        # ── Noise ─────────────────────────────────────────────────────────
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ISONoise(color_shift=(0.01, 0.05),
                       intensity=(0.1, 0.5)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
        ], p=aug.get('gaussian_noise', 0.2)),

        # ── Occlusion simulation ──────────────────────────────────────────
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=(114, 114, 114),
            p=aug.get('coarse_dropout', 0.3)
        ),

        A.GridDistortion(num_steps=5, distort_limit=0.2,
                         p=aug.get('grid_distortion', 0.2)),

        # ── JPEG compression artifact simulation ──────────────────────────
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2),

        # ── Weather simulation ────────────────────────────────────────────
        A.OneOf([
            A.RandomRain(slant_lower=-10, slant_upper=10,
                         drop_length=15, drop_width=1,
                         drop_color=(200, 200, 200), blur_value=2,
                         brightness_coefficient=0.9, p=1.0),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=1.0),
        ], p=0.15),

        # ── Normalize (always last) ───────────────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return A.Compose(transforms, bbox_params=bbox_params)


# ─────────────────────────────────────────────────────────────────────────────
# MOSAIC AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

class MosaicAugmentation:
    """
    Combines 4 images into a 2×2 mosaic for dense training.
    Improves detection of small damage regions.
    """

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
        self.h, self.w = target_size

    def __call__(self, images: List[np.ndarray],
                 boxes_list: List[np.ndarray],
                 classes_list: List[List[int]]) -> Tuple:
        """
        Args:
            images:       list of 4 preprocessed images [H, W, 3]
            boxes_list:   list of 4 box arrays [N, 4] in XYXY format
            classes_list: list of 4 class-id lists
        Returns:
            mosaic_image, combined_boxes, combined_classes
        """
        assert len(images) == 4, "Mosaic requires exactly 4 images"

        # Random center point
        cx = int(np.random.uniform(0.25, 0.75) * self.w)
        cy = int(np.random.uniform(0.25, 0.75) * self.h)

        mosaic = np.full((self.h, self.w, 3), 114, dtype=np.uint8)
        all_boxes, all_classes = [], []

        placements = [
            (0, 0, cx, cy),          # top-left
            (cx, 0, self.w, cy),     # top-right
            (0, cy, cx, self.h),     # bottom-left
            (cx, cy, self.w, self.h) # bottom-right
        ]

        for (x1, y1, x2, y2), img, boxes, classes in zip(
            placements, images, boxes_list, classes_list
        ):
            ph, pw = y2 - y1, x2 - x1
            img_resized = cv2.resize(img, (pw, ph))
            mosaic[y1:y2, x1:x2] = img_resized

            if len(boxes) > 0:
                scale_x = pw / img.shape[1]
                scale_y = ph / img.shape[0]

                adj_boxes = boxes.copy().astype(np.float32)
                adj_boxes[:, [0, 2]] = adj_boxes[:, [0, 2]] * scale_x + x1
                adj_boxes[:, [1, 3]] = adj_boxes[:, [1, 3]] * scale_y + y1
                adj_boxes[:, [0, 2]] = np.clip(adj_boxes[:, [0, 2]], x1, x2)
                adj_boxes[:, [1, 3]] = np.clip(adj_boxes[:, [1, 3]], y1, y2)

                # Remove boxes that are too small after clipping
                w_box = adj_boxes[:, 2] - adj_boxes[:, 0]
                h_box = adj_boxes[:, 3] - adj_boxes[:, 1]
                valid = (w_box > 4) & (h_box > 4)

                all_boxes.append(adj_boxes[valid])
                all_classes.extend([c for c, v in zip(classes, valid) if v])

        combined_boxes = np.concatenate(all_boxes, axis=0) \
            if all_boxes else np.zeros((0, 4))

        return mosaic, combined_boxes, all_classes
