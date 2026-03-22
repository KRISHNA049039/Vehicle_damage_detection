"""
Vehicle Damage Detection Model Package
Includes: Backbone, FPN, Detection/Segmentation/Severity Heads, Full Detector
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# ANCHOR GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class AnchorGenerator:
    """Generates multi-scale anchors for the FPN pyramid levels."""

    def __init__(self, sizes, ratios, scales, strides):
        self.sizes = sizes        # [32, 64, 128, 256, 512]
        self.ratios = ratios      # [0.5, 1.0, 2.0]
        self.scales = scales      # [1.0, 1.26, 1.587]
        self.strides = strides    # [8, 16, 32, 64, 128] for P3-P7

    def generate_anchors(self, image_shape):
        """Generate all anchors for the given image shape."""
        all_anchors = []
        h, w = image_shape

        for stride, size in zip(self.strides, self.sizes):
            grid_h = int(np.ceil(h / stride))
            grid_w = int(np.ceil(w / stride))
            anchors = self._generate_level_anchors(
                size, grid_h, grid_w, stride
            )
            all_anchors.append(anchors)

        return tf.concat(all_anchors, axis=0)  # [total_anchors, 4]

    def _generate_level_anchors(self, base_size, grid_h, grid_w, stride):
        anchors = []
        for scale in self.scales:
            for ratio in self.ratios:
                area = (base_size * scale) ** 2
                w = np.sqrt(area / ratio)
                h = ratio * w

                for gy in range(grid_h):
                    for gx in range(grid_w):
                        cx = (gx + 0.5) * stride
                        cy = (gy + 0.5) * stride
                        anchors.append([
                            cx - w / 2, cy - h / 2,
                            cx + w / 2, cy + h / 2
                        ])

        return tf.constant(anchors, dtype=tf.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE PYRAMID NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class FeaturePyramidNetwork(layers.Layer):
    """
    Standard FPN that takes multi-scale backbone features
    and produces P3-P7 feature maps at uniform channel depth.
    """

    def __init__(self, out_channels=256, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels

        # Lateral 1×1 projections
        self.lat_c3 = layers.Conv2D(out_channels, 1, padding='same',
                                    kernel_initializer='he_normal')
        self.lat_c4 = layers.Conv2D(out_channels, 1, padding='same',
                                    kernel_initializer='he_normal')
        self.lat_c5 = layers.Conv2D(out_channels, 1, padding='same',
                                    kernel_initializer='he_normal')

        # Top-down smoothing 3×3
        self.smooth_p3 = layers.Conv2D(out_channels, 3, padding='same',
                                       kernel_initializer='he_normal')
        self.smooth_p4 = layers.Conv2D(out_channels, 3, padding='same',
                                       kernel_initializer='he_normal')
        self.smooth_p5 = layers.Conv2D(out_channels, 3, padding='same',
                                       kernel_initializer='he_normal')

        # Extra pyramid levels P6, P7
        self.p6_conv = layers.Conv2D(out_channels, 3, strides=2, padding='same',
                                     kernel_initializer='he_normal')
        self.p7_relu = layers.ReLU()
        self.p7_conv = layers.Conv2D(out_channels, 3, strides=2, padding='same',
                                     kernel_initializer='he_normal')

        # BN for each level
        self.bns = [layers.BatchNormalization() for _ in range(5)]

    def call(self, features, training=False):
        c3, c4, c5 = features  # backbone outputs

        # Lateral connections
        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + tf.image.resize(p5, tf.shape(c4)[1:3])
        p3 = self.lat_c3(c3) + tf.image.resize(p4, tf.shape(c3)[1:3])

        # Smooth
        p3 = self.bns[0](self.smooth_p3(p3), training=training)
        p4 = self.bns[1](self.smooth_p4(p4), training=training)
        p5 = self.bns[2](self.smooth_p5(p5), training=training)
        p6 = self.bns[3](self.p6_conv(c5), training=training)
        p7 = self.bns[4](self.p7_conv(self.p7_relu(p6)), training=training)

        return [p3, p4, p5, p6, p7]


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION HEADS
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(layers.Layer):
    """
    Shared classification head applied across all FPN levels.
    Predicts objectness + class probabilities per anchor.
    """

    def __init__(self, num_classes, num_anchors, num_convs=4,
                 feat_channels=256, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.convs = []
        self.bns = []
        for i in range(num_convs):
            self.convs.append(
                layers.Conv2D(feat_channels, 3, padding='same',
                              activation='relu',
                              kernel_initializer='he_normal',
                              bias_initializer='zeros')
            )
            self.bns.append(layers.BatchNormalization())

        # Output: num_anchors * num_classes scores per spatial location
        import math
        bias_init = tf.constant_initializer(
            -math.log((1 - 0.01) / 0.01)
        )
        self.cls_conv = layers.Conv2D(
            num_anchors * num_classes, 3, padding='same',
            bias_initializer=bias_init
        )

    def call(self, x, training=False):
        for conv, bn in zip(self.convs, self.bns):
            x = bn(conv(x), training=training)
        out = self.cls_conv(x)
        # Reshape: [B, H, W, A*C] -> [B, H*W*A, C]
        B = tf.shape(out)[0]
        H = tf.shape(out)[1]
        W = tf.shape(out)[2]
        out = tf.reshape(out, [B, H * W * self.num_anchors, self.num_classes])
        return tf.sigmoid(out)


class RegressionHead(layers.Layer):
    """
    Shared box regression head applied across all FPN levels.
    Predicts (dx, dy, dw, dh) deltas per anchor.
    """

    def __init__(self, num_anchors, num_convs=4, feat_channels=256, **kwargs):
        super().__init__(**kwargs)
        self.num_anchors = num_anchors

        self.convs = []
        self.bns = []
        for _ in range(num_convs):
            self.convs.append(
                layers.Conv2D(feat_channels, 3, padding='same',
                              activation='relu',
                              kernel_initializer='he_normal')
            )
            self.bns.append(layers.BatchNormalization())

        self.box_conv = layers.Conv2D(num_anchors * 4, 3, padding='same')

    def call(self, x, training=False):
        for conv, bn in zip(self.convs, self.bns):
            x = bn(conv(x), training=training)
        out = self.box_conv(x)
        B = tf.shape(out)[0]
        H = tf.shape(out)[1]
        W = tf.shape(out)[2]
        out = tf.reshape(out, [B, H * W * self.num_anchors, 4])
        return out


# ─────────────────────────────────────────────────────────────────────────────
# MASK HEAD
# ─────────────────────────────────────────────────────────────────────────────

class MaskHead(layers.Layer):
    """
    FCN mask head: takes ROI-aligned features [7×7×256]
    and predicts binary masks [28×28] per class.
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

        self.convs = [
            layers.Conv2D(256, 3, padding='same', activation='relu')
            for _ in range(4)
        ]
        self.deconv = layers.Conv2DTranspose(256, 2, strides=2, activation='relu')
        self.mask_out = layers.Conv2D(num_classes, 1, activation='sigmoid')

    def call(self, x, training=False):
        for conv in self.convs:
            x = conv(x)
        x = self.deconv(x)
        return self.mask_out(x)  # [B, 28, 28, num_classes]


# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY HEAD
# ─────────────────────────────────────────────────────────────────────────────

class SeverityHead(layers.Layer):
    """
    Regresses severity score [0,1] per detected damage region.
    Also predicts the fractional area of damage for quantification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(512, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn1 = layers.BatchNormalization()
        self.drop1 = layers.Dropout(0.4)
        self.fc2 = layers.Dense(256, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn2 = layers.BatchNormalization()
        self.drop2 = layers.Dropout(0.3)
        self.severity_out = layers.Dense(1, activation='sigmoid',
                                         name='severity_score')
        self.area_out = layers.Dense(1, activation='sigmoid',
                                      name='area_fraction')
        self.occlusion_out = layers.Dense(1, activation='sigmoid',
                                           name='occlusion_level')

    def call(self, x, training=False):
        x = self.gap(x)
        x = self.drop1(self.bn1(self.fc1(x), training=training),
                        training=training)
        feat = self.drop2(self.bn2(self.fc2(x), training=training),
                           training=training)
        return {
            'severity': self.severity_out(feat),
            'area_fraction': self.area_out(feat),
            'occlusion': self.occlusion_out(feat)
        }


# ─────────────────────────────────────────────────────────────────────────────
# OCCLUSION-AWARE ATTENTION MODULE
# ─────────────────────────────────────────────────────────────────────────────

class OcclusionAwareAttention(layers.Layer):
    """
    Channel + spatial attention that down-weights occluded / reflected regions.
    Applied to each FPN level feature map.
    """

    def __init__(self, channels, reduction=16, **kwargs):
        super().__init__(**kwargs)
        # Channel attention (SE-style)
        self.gap_ch = layers.GlobalAveragePooling2D()
        self.fc_ch1 = layers.Dense(channels // reduction, activation='relu')
        self.fc_ch2 = layers.Dense(channels, activation='sigmoid')

        # Spatial attention
        self.conv_sp = layers.Conv2D(1, 7, padding='same', activation='sigmoid')

    def call(self, x, training=False):
        # Channel attention
        ch_att = self.gap_ch(x)
        ch_att = self.fc_ch2(self.fc_ch1(ch_att))
        ch_att = tf.reshape(ch_att, [-1, 1, 1, tf.shape(x)[-1]])
        x_ch = x * ch_att

        # Spatial attention
        sp_avg = tf.reduce_mean(x_ch, axis=-1, keepdims=True)
        sp_max = tf.reduce_max(x_ch, axis=-1, keepdims=True)
        sp_cat = tf.concat([sp_avg, sp_max], axis=-1)
        sp_att = self.conv_sp(sp_cat)
        return x_ch * sp_att


# ─────────────────────────────────────────────────────────────────────────────
# REFLECTION SUPPRESSION MODULE
# ─────────────────────────────────────────────────────────────────────────────

class ReflectionSuppressionModule(layers.Layer):
    """
    Learnable suppression of specular highlights in feature space.
    Detects high-activation outlier regions and normalizes them.
    """

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.gate_conv = layers.Conv2D(channels, 1, activation='sigmoid')
        self.refine_conv = layers.Conv2D(channels, 3, padding='same',
                                          activation='relu')
        self.bn = layers.BatchNormalization()

    def call(self, x, training=False):
        # Gate high-intensity channels
        gate = self.gate_conv(x)
        x_gated = x * gate
        x_refined = self.bn(self.refine_conv(x_gated), training=training)
        return x + x_refined  # residual


# ─────────────────────────────────────────────────────────────────────────────
# FULL DETECTOR MODEL
# ─────────────────────────────────────────────────────────────────────────────

class VehicleDamageDetector(Model):
    """
    Full multi-task vehicle damage detection model.

    Tasks:
      1. Object detection (bounding boxes + class labels)
      2. Instance segmentation (binary masks per detection)
      3. Severity regression (score 0→1 per damage region)
      4. Occlusion estimation (how much is hidden)

    Architecture:
      EfficientNet-B4 backbone → FPN (P3-P7)
      → Classification Head (RetinaNet-style)
      → Regression Head
      → Mask Head (Mask R-CNN style)
      → Severity Head
    """

    def __init__(self, num_classes=6, num_anchors=9,
                 fpn_channels=256, freeze_layers=100, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # ── Backbone ───────────────────────────────────────────────────────
        base = tf.keras.applications.EfficientNetB4(
            include_top=False,
            weights='imagenet',
            input_shape=(640, 640, 3)
        )
        # Freeze early layers for transfer learning
        for layer in base.layers[:freeze_layers]:
            layer.trainable = False

        # Extract C3, C4, C5 feature maps
        c3_out = base.get_layer('block3b_add').output     # stride 8
        c4_out = base.get_layer('block5d_add').output     # stride 16
        c5_out = base.get_layer('block7a_project_bn').output  # stride 32

        self.backbone = tf.keras.Model(
            inputs=base.input,
            outputs=[c3_out, c4_out, c5_out]
        )

        # ── FPN ────────────────────────────────────────────────────────────
        self.fpn = FeaturePyramidNetwork(out_channels=fpn_channels)

        # ── Attention & Reflection Modules ─────────────────────────────────
        self.attention_modules = [
            OcclusionAwareAttention(fpn_channels) for _ in range(5)
        ]
        self.reflection_modules = [
            ReflectionSuppressionModule(fpn_channels) for _ in range(5)
        ]

        # ── Detection Heads ────────────────────────────────────────────────
        self.cls_head = ClassificationHead(num_classes, num_anchors)
        self.box_head = RegressionHead(num_anchors)

        # ── Segmentation Head ──────────────────────────────────────────────
        self.mask_head = MaskHead(num_classes)

        # ── Severity Head ──────────────────────────────────────────────────
        self.severity_head = SeverityHead()

        # ── Upsampling for mask ────────────────────────────────────────────
        self.roi_pool = layers.GlobalAveragePooling2D()
        self.roi_upsample = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')

    def call(self, inputs, training=False):
        # 1. Backbone: multi-scale features
        c3, c4, c5 = self.backbone(inputs, training=training)

        # 2. FPN: P3–P7
        pyramid = self.fpn([c3, c4, c5], training=training)

        # 3. Apply attention + reflection suppression per level
        enhanced = []
        for i, (p, att, ref) in enumerate(
            zip(pyramid, self.attention_modules, self.reflection_modules)
        ):
            p = att(p, training=training)
            p = ref(p, training=training)
            enhanced.append(p)

        # 4. Detection outputs (applied to each pyramid level)
        cls_outputs = [self.cls_head(p, training=training) for p in enhanced]
        box_outputs = [self.box_head(p, training=training) for p in enhanced]

        # 5. Segmentation (use P3 — highest resolution)
        mask_outputs = self.mask_head(enhanced[0], training=training)
        # Upsample masks to 1/4 input resolution
        mask_outputs = self.roi_upsample(mask_outputs)

        # 6. Severity estimation (use P4)
        severity_outputs = self.severity_head(enhanced[1], training=training)

        return {
            'cls': cls_outputs,       # list of [B, H*W*A, C] per level
            'box': box_outputs,       # list of [B, H*W*A, 4] per level
            'masks': mask_outputs,    # [B, H/4, W/4, num_classes]
            'severity': severity_outputs['severity'],      # [B, 1]
            'area_fraction': severity_outputs['area_fraction'],  # [B, 1]
            'occlusion': severity_outputs['occlusion']     # [B, 1]
        }

    def build_graph(self, input_shape=(640, 640, 3)):
        """Helper to build model graph for summary."""
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> VehicleDamageDetector:
    """Factory function to build model from config dict."""
    model = VehicleDamageDetector(
        num_classes=cfg['model']['num_classes'],
        num_anchors=cfg['model']['num_anchors'],
        fpn_channels=cfg['model']['fpn_out_channels'],
        freeze_layers=cfg['model']['freeze_backbone_layers']
    )
    # Build with dummy input
    dummy = tf.zeros([1, *cfg['model']['image_size'], 3])
    _ = model(dummy, training=False)
    return model
