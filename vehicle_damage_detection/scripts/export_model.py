"""
Model Export Utilities

Exports trained model to:
  - TensorFlow SavedModel (TF Serving)
  - TFLite (mobile / edge)
  - ONNX (cross-framework inference)

Usage:
    python scripts/export_model.py \
        --checkpoint checkpoints/best_model \
        --output-dir exports/ \
        --formats saved_model tflite
"""

import os
import sys
import argparse
import yaml
import numpy as np
import tensorflow as tf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import build_model


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class DamageDetectorExportWrapper(tf.Module):
    """
    Wraps the full inference pipeline (preprocess → model → postprocess)
    into a single exportable TF function.

    This makes deployment simple: just call the exported function with
    a raw uint8 image tensor.
    """

    def __init__(self, model, image_size=(640, 640)):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.imagenet_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.imagenet_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    def _preprocess(self, image: tf.Tensor) -> tf.Tensor:
        """Resize, normalize — no letterboxing for fixed-shape export."""
        image = tf.image.resize(image, self.image_size)
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - self.imagenet_mean) / self.imagenet_std
        return image

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.uint8,
                       name='input_images')
    ])
    def predict(self, images: tf.Tensor):
        """
        Full inference pipeline.

        Input:  uint8 RGB images [B, H, W, 3]
        Output: dict with cls, box, masks, severity, occlusion
        """
        preprocessed = tf.map_fn(
            self._preprocess, images,
            fn_output_signature=tf.float32
        )
        return self.model(preprocessed, training=False)


# ─────────────────────────────────────────────────────────────────────────────
# SAVEDMODEL EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_saved_model(model, output_dir: str,
                        image_size: tuple = (640, 640)):
    """
    Export to TensorFlow SavedModel format.
    Compatible with TF Serving and TF.js.

    Args:
        model:       trained VehicleDamageDetector
        output_dir:  directory to save the model
        image_size:  inference input size
    """
    output_path = Path(output_dir) / 'saved_model'
    output_path.mkdir(parents=True, exist_ok=True)

    wrapper = DamageDetectorExportWrapper(model, image_size)

    # Warm up to trace graph
    dummy = tf.zeros([1, *image_size, 3], dtype=tf.uint8)
    _ = wrapper.predict(dummy)

    tf.saved_model.save(
        wrapper, str(output_path),
        signatures={'serving_default': wrapper.predict}
    )
    print(f"SavedModel exported to: {output_path}")
    return str(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# TFLITE EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_tflite(saved_model_path: str, output_dir: str,
                   quantize: bool = True,
                   representative_data: list = None):
    """
    Export to TFLite format for mobile/edge deployment.

    Args:
        saved_model_path:    path to TF SavedModel
        output_dir:          where to save .tflite file
        quantize:            apply INT8 post-training quantization
        representative_data: list of sample numpy arrays for quantization

    Quantization reduces model size by ~4x and speeds up inference
    on devices with INT8 accelerators (e.g. Edge TPU, ARM).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantize and representative_data is not None:
        def representative_dataset():
            for sample in representative_data[:100]:
                yield [sample.astype(np.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32
        tflite_path = output_path / 'vehicle_damage_int8.tflite'
    else:
        tflite_path = output_path / 'vehicle_damage_fp16.tflite'
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)

    size_mb = len(tflite_model) / (1024 ** 2)
    print(f"TFLite model exported: {tflite_path} ({size_mb:.1f} MB)")
    return str(tflite_path)


# ─────────────────────────────────────────────────────────────────────────────
# TFLITE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

class TFLiteInferenceEngine:
    """
    Runs inference using a .tflite model.
    Useful for edge deployment without full TF installation.
    """

    def __init__(self, tflite_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]['shape']
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]
        print(f"TFLite engine ready: input shape {input_shape}")

    def predict(self, image: np.ndarray) -> dict:
        """
        Args:
            image: BGR uint8 [H, W, 3]
        Returns:
            dict of output tensors
        """
        import cv2

        # Preprocess
        resized = cv2.resize(image, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb, 0).astype(np.uint8)

        # Set input
        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_data
        )

        # Run
        self.interpreter.invoke()

        # Collect outputs
        outputs = {}
        for detail in self.output_details:
            outputs[detail['name']] = self.interpreter.get_tensor(
                detail['index']
            )

        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# ONNX EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(saved_model_path: str, output_dir: str,
                opset: int = 13):
    """
    Export to ONNX format.
    Requires: pip install tf2onnx onnx onnxruntime

    ONNX enables deployment in:
      - PyTorch ecosystem
      - TensorRT (NVIDIA GPU optimized)
      - OpenVINO (Intel CPU/VPU)
      - ONNX Runtime (cross-platform)
    """
    try:
        import tf2onnx
        import onnx
    except ImportError:
        print("ONNX export requires: pip install tf2onnx onnx")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_path = str(output_path / 'vehicle_damage.onnx')

    model_proto, _ = tf2onnx.convert.from_saved_model(
        saved_model_path,
        opset=opset,
        output_path=onnx_path
    )

    # Validate ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    size_mb = Path(onnx_path).stat().st_size / (1024 ** 2)
    print(f"ONNX model exported: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BENCHMARKING
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_model(model_or_engine, n_runs: int = 50,
                     image_size: tuple = (640, 640),
                     batch_size: int = 1) -> dict:
    """
    Benchmark model inference speed and memory.

    Args:
        model_or_engine: TF model or TFLiteInferenceEngine
        n_runs:          number of warmup + timed runs
        image_size:      input image size
        batch_size:      batch size to test
    Returns:
        dict with latency statistics
    """
    import time

    dummy = np.random.randint(
        0, 255, (batch_size, *image_size, 3), dtype=np.uint8
    )

    # Warmup
    print(f"Warming up ({5} runs)...")
    for _ in range(5):
        if isinstance(model_or_engine, TFLiteInferenceEngine):
            model_or_engine.predict(dummy[0])
        else:
            _ = model_or_engine(
                tf.constant(dummy / 255.0, dtype=tf.float32),
                training=False
            )

    # Timed runs
    print(f"Benchmarking ({n_runs} runs)...")
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        if isinstance(model_or_engine, TFLiteInferenceEngine):
            model_or_engine.predict(dummy[0])
        else:
            _ = model_or_engine(
                tf.constant(dummy / 255.0, dtype=tf.float32),
                training=False
            )
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    results = {
        'batch_size': batch_size,
        'image_size': image_size,
        'n_runs': n_runs,
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'fps': float(batch_size / (np.mean(latencies) / 1000))
    }

    print(f"\nBenchmark Results:")
    print(f"  Mean latency: {results['mean_ms']:.1f} ms")
    print(f"  P95  latency: {results['p95_ms']:.1f} ms")
    print(f"  Throughput:   {results['fps']:.1f} FPS")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export vehicle damage detection model"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint weights')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--output-dir', type=str, default='exports/')
    parser.add_argument('--formats', nargs='+',
                        choices=['saved_model', 'tflite', 'onnx'],
                        default=['saved_model', 'tflite'],
                        help='Export formats')
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--quantize', action='store_true',
                        help='Apply INT8 quantization to TFLite export')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run inference benchmark after export')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    image_size = (args.image_size, args.image_size)

    print("Loading model...")
    model = build_model(cfg)
    model.load_weights(args.checkpoint)
    print("Model loaded.")

    saved_model_path = None

    if 'saved_model' in args.formats or 'tflite' in args.formats \
            or 'onnx' in args.formats:
        saved_model_path = export_saved_model(model, args.output_dir,
                                               image_size)

    if 'tflite' in args.formats:
        export_tflite(saved_model_path, args.output_dir,
                       quantize=args.quantize)

    if 'onnx' in args.formats:
        export_onnx(saved_model_path, args.output_dir)

    if args.benchmark:
        print("\nRunning benchmark...")
        benchmark_model(model, n_runs=50, image_size=image_size)

    print(f"\nAll exports saved to: {args.output_dir}")
