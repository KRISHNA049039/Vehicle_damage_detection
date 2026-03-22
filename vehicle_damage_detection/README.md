# Vehicle Damage Detection System

End-to-end deep learning system for automated vehicle damage detection, segmentation, and severity estimation using TensorFlow 2.x.

---

## Project Structure

```
vehicle_damage_detection/
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── src/
│   ├── model/
│   │   ├── __init__.py          # Full detector model (backbone, FPN, heads)
│   │   └── losses.py            # Focal, Smooth L1, Dice, Huber losses
│   ├── data/
│   │   ├── dataset.py           # COCO loader, TFRecord writer, tf.data pipeline
│   │   └── preprocessing.py     # CLAHE, highlight suppression, augmentation
│   ├── evaluation/
│   │   └── metrics.py           # mAP, mIoU, severity MAE, Pearson R
│   ├── utils/
│   │   └── postprocessing.py    # NMS, mask extraction, damage report
│   └── api/
│       └── app.py               # FastAPI REST API
├── scripts/
│   └── prepare_data.py          # COCO → TFRecord conversion
├── tests/
│   └── test_all.py              # Unit and integration tests
├── train.py                     # Training entry point
├── evaluate.py                  # Evaluation entry point
├── predict.py                   # Inference entry point
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Setup

```bash
# 1. Clone and create environment
git clone <repo>
cd vehicle_damage_detection
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data (COCO format annotations required)
python scripts/prepare_data.py \
    --train-annotations data/annotations/train.json \
    --train-images data/images/train \
    --val-annotations data/annotations/val.json \
    --val-images data/images/val \
    --output-dir data/tfrecords
```

---

## Training

```bash
# Standard training
python train.py --config configs/config.yaml

# Resume from checkpoint
python train.py --config configs/config.yaml --resume checkpoints/ckpt-50

# Debug mode (small dataset)
python train.py --config configs/config.yaml --debug --no-wandb
```

---

## Evaluation

```bash
# Evaluate on test set
python evaluate.py --model checkpoints/best_model --split test

# Save predictions
python evaluate.py --model checkpoints/best_model --split test --save-predictions
```

---

## Inference

```bash
# Single image
python predict.py --image car.jpg --model checkpoints/best_model

# Batch directory
python predict.py --input-dir images/ --model checkpoints/best_model --output-dir results/

# Custom thresholds
python predict.py --image car.jpg --model checkpoints/best_model \
    --threshold 0.5 --nms-threshold 0.5
```

---

## REST API

```bash
# Start server
uvicorn src.api.app:app --host 0.0.0.0 --port 8080

# Or with Docker
docker build -t vehicle-damage .
docker run -p 8080:8080 vehicle-damage

# API docs: http://localhost:8080/docs
```

### API Usage

```python
import requests

# Single image prediction
with open('car.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8080/predict',
        files={'file': ('car.jpg', f, 'image/jpeg')},
        params={'include_annotated_image': True}
    )

result = response.json()
print(f"Overall severity: {result['damage_report']['vehicle_overall_label']}")
print(f"Estimated cost: ${result['damage_report']['cost_estimate_usd']['midpoint']}")
```

---

## Damage Classes

| Class | Description |
|-------|-------------|
| `scratch` | Linear paint abrasion |
| `dent` | Sheet metal depression |
| `glass_damage` | Cracks, chips in glass |
| `tear` | Rips in bumper/trim |
| `body_deform` | Large panel deformation |

## Severity Scale

| Label | Score Range | Description |
|-------|-------------|-------------|
| MINOR | 0.00 – 0.30 | Surface/cosmetic only |
| MODERATE | 0.30 – 0.65 | Repair needed |
| SEVERE | 0.65 – 1.00 | Safety concern, major repair |

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run fast tests only (skip slow model tests)
pytest tests/ -v -m "not slow"
```

---

## Key Design Decisions

- **EfficientNet-B4 backbone**: Best accuracy/efficiency trade-off for vehicle-scale images
- **FPN (P3–P7)**: Essential for detecting both small scratches and large body damage
- **Focal Loss**: Handles severe foreground/background class imbalance
- **CLAHE preprocessing**: Dramatically improves scratch visibility on low-contrast panels
- **Specular highlight suppression**: Prevents metallic reflections from causing false positives
- **Occlusion-aware attention**: Down-weights features in occluded/shadow regions
- **Severity regression**: Continuous score enables fine-grained damage cost estimation
