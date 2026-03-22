"""
FastAPI REST API for Vehicle Damage Detection

Endpoints:
  POST /predict         - Single image analysis
  POST /predict/batch   - Multiple images (up to 10)
  GET  /health          - Health check
  GET  /classes         - List damage classes
  GET  /docs            - Auto-generated Swagger UI

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8080
    # or
    python -m src.api.app
"""

import os
import io
import sys
import json
import base64
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int


class DamageRegion(BaseModel):
    id: int
    cls: str = Field(alias='class')
    confidence: float
    bbox: BoundingBox
    mask_area_px: int
    mask_area_pct: float
    severity_score: float
    severity_label: str
    occlusion_level: float
    confidence_adjusted: float
    cost_estimate_usd: Dict[str, int]

    class Config:
        populate_by_name = True


class DamageByClass(BaseModel):
    count: int
    total_area_pct: float
    mean_severity: float


class CostEstimate(BaseModel):
    low: int
    high: int
    midpoint: int


class DamageReport(BaseModel):
    vehicle_overall_severity: float
    vehicle_overall_label: str
    total_damage_area_pct: float
    damage_count: int
    damage_by_class: Dict[str, DamageByClass]
    cost_estimate_usd: CostEstimate
    regions: List[Dict]
    image_dimensions: Dict[str, int]


class PredictResponse(BaseModel):
    success: bool
    processing_time_ms: float
    damage_report: Dict[str, Any]
    annotated_image_base64: Optional[str] = None
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    timestamp: str


class ClassesResponse(BaseModel):
    classes: List[str]
    descriptions: Dict[str, str]
    severity_bands: Dict[str, Dict[str, float]]


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def create_app(config_path: str = 'configs/config.yaml',
               model_path: str = 'checkpoints/best_model') -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Vehicle Damage Detection API",
        description=(
            "Automated vehicle damage assessment using deep learning. "
            "Detects scratches, dents, glass damage, tears, and body deformation. "
            "Returns bounding boxes, segmentation masks, severity scores, and cost estimates."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # ── Load config ─────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── Load model lazily ────────────────────────────────────────────────
    app.state.engine = None
    app.state.model_path = model_path
    app.state.config_path = config_path
    app.state.cfg = cfg

    @app.on_event("startup")
    async def startup():
        """Load model on startup."""
        try:
            from predict import VehicleDamageInferenceEngine
            app.state.engine = VehicleDamageInferenceEngine(
                model_path=app.state.model_path,
                config_path=app.state.config_path,
                score_threshold=cfg['inference']['score_threshold'],
                nms_iou_threshold=cfg['inference']['nms_iou_threshold'],
                max_detections=cfg['inference']['max_detections']
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Running in demo mode")

    # ── Utility functions ────────────────────────────────────────────────

    def decode_image(file_bytes: bytes) -> np.ndarray:
        """Decode uploaded image bytes to numpy array."""
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Supported: JPEG, PNG, BMP"
            )
        return image

    def encode_image(image: np.ndarray) -> str:
        """Encode BGR image to base64 JPEG string."""
        _, buffer = cv2.imencode('.jpg', image,
                                  [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def make_demo_report(h: int, w: int) -> Dict:
        """Generate a demo report when model is not loaded."""
        return {
            'vehicle_overall_severity': 0.42,
            'vehicle_overall_label': 'MODERATE',
            'total_damage_area_pct': 5.2,
            'damage_count': 2,
            'damage_by_class': {
                'scratch': {'count': 2, 'total_area_pct': 3.1, 'mean_severity': 0.38},
                'dent': {'count': 0, 'total_area_pct': 0.0, 'mean_severity': 0.0},
                'glass_damage': {'count': 0, 'total_area_pct': 0.0, 'mean_severity': 0.0},
                'tear': {'count': 0, 'total_area_pct': 0.0, 'mean_severity': 0.0},
                'body_deform': {'count': 0, 'total_area_pct': 0.0, 'mean_severity': 0.0},
            },
            'cost_estimate_usd': {'low': 150, 'high': 600, 'midpoint': 375},
            'regions': [
                {
                    'id': 0, 'class': 'scratch', 'confidence': 0.87,
                    'bbox': {'x1': 175, 'y1': 510, 'x2': 420, 'y2': 620,
                             'width': 245, 'height': 110},
                    'mask_area_px': 8420, 'mask_area_pct': 1.42,
                    'severity_score': 0.34, 'severity_label': 'MODERATE',
                    'occlusion_level': 0.05, 'confidence_adjusted': 0.858,
                    'cost_estimate_usd': {'low': 50, 'high': 200}
                }
            ],
            'image_dimensions': {'height': h, 'width': w},
            '_demo_mode': True
        }

    # ── ENDPOINTS ────────────────────────────────────────────────────────

    @app.get('/health', response_model=HealthResponse, tags=['System'])
    async def health():
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'model_loaded': app.state.engine is not None,
            'version': '1.0.0',
            'timestamp': datetime.utcnow().isoformat()
        }

    @app.get('/classes', response_model=ClassesResponse, tags=['System'])
    async def get_classes():
        """List all damage classes with descriptions."""
        return {
            'classes': [
                'background', 'scratch', 'dent',
                'glass_damage', 'tear', 'body_deform'
            ],
            'descriptions': {
                'scratch': 'Linear paint abrasion on vehicle surface',
                'dent': 'Sheet metal depression or deformation',
                'glass_damage': 'Cracks, chips or fractures in windshield/windows',
                'tear': 'Rips in bumper, trim, or soft body parts',
                'body_deform': 'Large crumple zones, creases, severe panel damage'
            },
            'severity_bands': {
                'minor':    {'min': 0.00, 'max': 0.30},
                'moderate': {'min': 0.30, 'max': 0.65},
                'severe':   {'min': 0.65, 'max': 1.00}
            }
        }

    @app.post('/predict', response_model=PredictResponse, tags=['Prediction'])
    async def predict(
        file: UploadFile = File(..., description="Vehicle image (JPEG/PNG)"),
        include_annotated_image: bool = True,
    ):
        """
        Analyze a single vehicle image for damage.

        Returns:
          - Damage report with detections, severity scores, cost estimates
          - Optionally: base64-encoded annotated image
        """
        # Validate file type
        if file.content_type not in [
            'image/jpeg', 'image/jpg', 'image/png', 'image/bmp'
        ]:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported media type: {file.content_type}"
            )

        file_bytes = await file.read()
        image = decode_image(file_bytes)
        orig_h, orig_w = image.shape[:2]

        t_start = time.time()

        if app.state.engine is not None:
            # Real model inference
            annotated, report = app.state.engine.predict(image)
        else:
            # Demo mode
            annotated = image
            report = make_demo_report(orig_h, orig_w)

        elapsed_ms = (time.time() - t_start) * 1000

        response = {
            'success': True,
            'processing_time_ms': round(elapsed_ms, 2),
            'damage_report': report,
            'timestamp': datetime.utcnow().isoformat()
        }

        if include_annotated_image:
            response['annotated_image_base64'] = encode_image(annotated)

        return response

    @app.post('/predict/batch', tags=['Prediction'])
    async def predict_batch(
        files: List[UploadFile] = File(..., description="Up to 10 vehicle images"),
        include_annotated_images: bool = False,
    ):
        """
        Analyze multiple vehicle images in a single request.

        Maximum 10 images per request.
        """
        if len(files) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 images per batch request"
            )

        results = []
        t_batch_start = time.time()

        for file in files:
            file_bytes = await file.read()
            try:
                image = decode_image(file_bytes)
                orig_h, orig_w = image.shape[:2]

                t_start = time.time()

                if app.state.engine is not None:
                    annotated, report = app.state.engine.predict(image)
                else:
                    annotated = image
                    report = make_demo_report(orig_h, orig_w)

                elapsed_ms = (time.time() - t_start) * 1000

                result = {
                    'filename': file.filename,
                    'success': True,
                    'processing_time_ms': round(elapsed_ms, 2),
                    'damage_report': report
                }

                if include_annotated_images:
                    result['annotated_image_base64'] = encode_image(annotated)

            except HTTPException as e:
                result = {
                    'filename': file.filename,
                    'success': False,
                    'error': e.detail
                }

            results.append(result)

        return {
            'batch_size': len(files),
            'total_processing_time_ms': round(
                (time.time() - t_batch_start) * 1000, 2
            ),
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        }

    @app.post('/predict/base64', tags=['Prediction'])
    async def predict_base64(payload: Dict[str, str]):
        """
        Analyze an image provided as base64-encoded string.

        Payload: {"image_base64": "<base64 string>"}
        """
        if 'image_base64' not in payload:
            raise HTTPException(
                status_code=400,
                detail="Missing 'image_base64' field"
            )

        try:
            img_bytes = base64.b64decode(payload['image_base64'])
            image = decode_image(img_bytes)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 image: {str(e)}"
            )

        orig_h, orig_w = image.shape[:2]
        t_start = time.time()

        if app.state.engine is not None:
            annotated, report = app.state.engine.predict(image)
        else:
            annotated = image
            report = make_demo_report(orig_h, orig_w)

        elapsed_ms = (time.time() - t_start) * 1000

        return {
            'success': True,
            'processing_time_ms': round(elapsed_ms, 2),
            'damage_report': report,
            'annotated_image_base64': encode_image(annotated),
            'timestamp': datetime.utcnow().isoformat()
        }

    return app


# ─────────────────────────────────────────────────────────────────────────────
# APP INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

app = create_app()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'src.api.app:app',
        host='0.0.0.0',
        port=8080,
        reload=False,
        workers=1
    )
