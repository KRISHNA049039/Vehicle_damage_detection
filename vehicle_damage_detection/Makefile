# Vehicle Damage Detection — Makefile

.PHONY: install prepare train evaluate predict api test export clean lint

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Data Preparation ────────────────────────────────────────────────────────
prepare:
	python scripts/prepare_data.py \
		--train-annotations data/annotations/train.json \
		--train-images data/images/train \
		--val-annotations data/annotations/val.json \
		--val-images data/images/val \
		--test-annotations data/annotations/test.json \
		--test-images data/images/test \
		--output-dir data/tfrecords

prepare-stats:
	python scripts/prepare_data.py \
		--train-annotations data/annotations/train.json \
		--train-images data/images/train \
		--val-annotations data/annotations/val.json \
		--val-images data/images/val \
		--output-dir data/tfrecords \
		--stats-only

# ── Training ────────────────────────────────────────────────────────────────
train:
	python train.py --config configs/config.yaml

train-debug:
	python train.py --config configs/config.yaml --debug --no-wandb

resume:
	python train.py --config configs/config.yaml \
		--resume checkpoints/ckpt-$(EPOCH)

# ── Evaluation ──────────────────────────────────────────────────────────────
evaluate:
	python evaluate.py \
		--model checkpoints/best_model \
		--split test \
		--save-predictions \
		--output-dir evaluation_results/

evaluate-val:
	python evaluate.py \
		--model checkpoints/best_model \
		--split val \
		--output-dir evaluation_results/

# ── Inference ────────────────────────────────────────────────────────────────
predict:
	python predict.py \
		--image $(IMAGE) \
		--model checkpoints/best_model \
		--output-dir results/

predict-batch:
	python predict.py \
		--input-dir $(DIR) \
		--model checkpoints/best_model \
		--output-dir results/

# ── API ──────────────────────────────────────────────────────────────────────
api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --reload

api-prod:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --workers 2

api-docker:
	docker-compose up api

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v -m "not slow" --tb=short

test-coverage:
	pytest tests/ -v --cov=src --cov-report=html --tb=short

# ── Export ────────────────────────────────────────────────────────────────────
export:
	python scripts/export_model.py \
		--checkpoint checkpoints/best_model \
		--output-dir exports/ \
		--formats saved_model tflite

export-quantized:
	python scripts/export_model.py \
		--checkpoint checkpoints/best_model \
		--output-dir exports/ \
		--formats tflite \
		--quantize

benchmark:
	python scripts/export_model.py \
		--checkpoint checkpoints/best_model \
		--output-dir exports/ \
		--formats saved_model \
		--benchmark

# ── Visualization ─────────────────────────────────────────────────────────────
viz-samples:
	python scripts/visualize.py \
		--mode samples \
		--annotation-file data/annotations/train.json \
		--image-dir data/images/train \
		--output-dir plots/

viz-stats:
	python scripts/visualize.py \
		--mode stats \
		--stats-file data/tfrecords/dataset_stats.json \
		--output-dir plots/

viz-eval:
	python scripts/visualize.py \
		--mode evaluation \
		--metrics-file evaluation_results/test_metrics.json \
		--output-dir plots/

tensorboard:
	tensorboard --logdir logs/ --port 6006

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	flake8 src/ train.py predict.py evaluate.py \
		--max-line-length 100 \
		--ignore E402,W503

format:
	black src/ train.py predict.py evaluate.py --line-length 100

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t vehicle-damage-detection:latest .

docker-run:
	docker-compose up api

docker-train:
	docker-compose --profile training up train

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage

clean-all: clean
	rm -rf checkpoints/ logs/ exports/ results/ evaluation_results/ plots/

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Vehicle Damage Detection — Available Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make install        Install Python dependencies"
	@echo "  make prepare        Convert COCO annotations to TFRecords"
	@echo "  make train          Run training"
	@echo "  make train-debug    Run debug training (small dataset)"
	@echo "  make evaluate       Evaluate on test set"
	@echo "  make predict IMAGE=car.jpg   Predict single image"
	@echo "  make predict-batch DIR=imgs/ Predict directory"
	@echo "  make api            Start FastAPI server (dev)"
	@echo "  make api-docker     Start via Docker Compose"
	@echo "  make test           Run unit tests"
	@echo "  make export         Export to SavedModel + TFLite"
	@echo "  make benchmark      Benchmark inference speed"
	@echo "  make viz-samples    Visualize training samples"
	@echo "  make viz-stats      Plot dataset statistics"
	@echo "  make viz-eval       Plot evaluation results"
	@echo "  make tensorboard    Launch TensorBoard"
	@echo "  make lint           Run flake8 linter"
	@echo "  make clean          Remove build artifacts"
	@echo ""
