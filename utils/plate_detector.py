"""
License plate detection with YOLO (plate.pt).
Runs on a full scene or a vehicle crop; returns the cropped plate and bbox in input coordinates.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from utils.config import PLATE_MODEL_PATH
from utils.draw import draw_bbox

ImageSource = Union[str, np.ndarray]

_model: Optional[YOLO] = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        if not PLATE_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Plate model not found: {PLATE_MODEL_PATH}. "
                "Place plate.pt there or set VISIONX_PLATE_MODEL."
            )
        _model = YOLO(str(PLATE_MODEL_PATH))
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


def _run_plate_yolo(
    bgr: np.ndarray,
    model: YOLO,
    *,
    pad_x_ratio: Optional[float] = None,
    pad_y_ratio: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    results = model(bgr)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_idx = int(boxes.conf.argmax().item())
    box = boxes.xyxy[best_idx].tolist()
    conf = float(boxes.conf[best_idx].item())
    x1, y1, x2, y2 = map(int, box)

    h, w = bgr.shape[:2]
    if pad_x_ratio is not None and pad_y_ratio is not None:
        pad_x = int((x2 - x1) * pad_x_ratio)
        pad_y = int((y2 - y1) * pad_y_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = bgr[y1:y2, x1:x2]
    bbox: List[int] = [x1, y1, x2, y2]
    label = f"plate {conf:.2f}"
    annotated = draw_bbox(bgr, bbox, label=label, color=(0, 165, 255))

    return {
        "crop": crop,
        "bbox": bbox,
        "confidence": conf,
        "annotated": annotated,
    }


def detect_plate(image: ImageSource) -> Optional[Dict[str, Any]]:
    """
    Detect the highest-confidence plate box on `image`.

    Returns None if no plate, otherwise:
      crop: BGR plate crop
      bbox: [x1, y1, x2, y2] relative to the same image you passed in
      confidence: float
      annotated: input image with the plate box drawn
    """
    bgr = _load_bgr(image)
    return _run_plate_yolo(bgr, _get_model())


def plate_bbox_in_full_image(
    vehicle_bbox: List[int], plate_bbox_in_crop: List[int]
) -> List[int]:
    """Map plate box from vehicle-crop coordinates to full-image coordinates."""
    vx1, vy1, _, _ = vehicle_bbox
    px1, py1, px2, py2 = plate_bbox_in_crop
    return [vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2]


class PlateDetector:
    """
    Notebook-friendly wrapper: load an explicit .pt path and return (image, plate_crop) tuples.
    Prefer module-level detect_plate() for the main pipeline.
    """

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect_plate(
        self, image_path: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        det = _run_plate_yolo(img, self.model)
        if det is None:
            return img, None
        return img, det["crop"]

    def detect_plate_with_padding(
        self,
        image_path: str,
        pad_x_ratio: float = 0.05,
        pad_y_ratio: float = 0.10,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        det = _run_plate_yolo(
            img,
            self.model,
            pad_x_ratio=pad_x_ratio,
            pad_y_ratio=pad_y_ratio,
        )
        if det is None:
            return img, None
        return img, det["crop"]
