"""
Vehicle detection with YOLO (car.pt).
Accepts an image path or a BGR numpy array; returns crop, bbox, confidence, and annotated view.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

from utils.config import VEHICLE_MODEL_PATH
from utils.draw import draw_bbox

ImageSource = Union[str, np.ndarray]

_model: Optional[YOLO] = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        if not VEHICLE_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Vehicle model not found: {VEHICLE_MODEL_PATH}. "
                "Place car.pt there or set VISIONX_VEHICLE_MODEL."
            )
        _model = YOLO(str(VEHICLE_MODEL_PATH))
    return _model


def _load_bgr(image: ImageSource) -> np.ndarray:
    if isinstance(image, str):
        if not os.path.isfile(image):
            raise FileNotFoundError(f"Image not found: {image}")
        arr = cv2.imread(image)
        if arr is None:
            raise ValueError(f"Could not read image: {image}")
        return arr
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a file path (str) or BGR numpy array")
    if image.size == 0:
        raise ValueError("empty image array")
    return image


def detect_vehicle(image: ImageSource) -> Optional[Dict[str, Any]]:
    """
    Detect the highest-confidence vehicle box.

    Returns None if nothing is found, otherwise:
      crop: BGR crop of the vehicle
      bbox: [x1, y1, x2, y2] in the same coordinate system as the input image
      confidence: float
      annotated: full input image with the box drawn
    """
    bgr = _load_bgr(image)
    results = _get_model()(bgr)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_idx = int(boxes.conf.argmax().item())
    box = boxes.xyxy[best_idx].tolist()
    conf = float(boxes.conf[best_idx].item())
    x1, y1, x2, y2 = map(int, box)

    h, w = bgr.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = bgr[y1:y2, x1:x2]
    bbox: List[int] = [x1, y1, x2, y2]
    label = f"vehicle {conf:.2f}"
    annotated = draw_bbox(bgr, bbox, label=label, color=(0, 200, 0))

    return {
        "crop": crop,
        "bbox": bbox,
        "confidence": conf,
        "annotated": annotated,
    }
