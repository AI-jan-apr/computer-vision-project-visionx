"""
Vehicle model + brand from a YOLO classification (or similar) checkpoint trained in Colab.
Maps coarse class name to manufacturer via BRAND_MAP (extend when you add classes).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from ultralytics import YOLO

from utils.config import BRAND_MODEL_PATH

_model: Optional[YOLO] = None

# Keys must match your dataset class names from training (lowercase).
BRAND_MAP: Dict[str, str] = {
    "accent": "Hyundai",
    "elentra": "Hyundai",
    "camry": "Toyota",
    "hilux": "Toyota",
    "patrol": "Nissan",
}


def _get_model() -> YOLO:
    global _model
    if _model is None:
        if not BRAND_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Brand/model classifier not found: {BRAND_MODEL_PATH}. "
                "Export best.pt from Colab into models/car_brand_model_detector/ "
                "or set VISIONX_BRAND_MODEL."
            )
        _model = YOLO(str(BRAND_MODEL_PATH))
    return _model


def predict_brand_model(vehicle_crop: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    Predict car model class and map to brand.

    Returns keys: model (str), brand (str), confidence (float).
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return {"model": "Unknown", "brand": "Unknown", "confidence": 0.0}

    results = _get_model().predict(vehicle_crop, verbose=False)
    probs = results[0].probs
    if probs is None:
        return {"model": "Unknown", "brand": "Unknown", "confidence": 0.0}

    class_id = int(probs.top1)
    class_name = results[0].names[class_id]
    confidence = float(probs.top1conf)
    key = str(class_name).lower().strip()
    brand = BRAND_MAP.get(key, "Unknown")

    return {
        "model": str(class_name),
        "brand": brand,
        "confidence": round(confidence, 4),
    }
