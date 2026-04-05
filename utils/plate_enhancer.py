"""
Preprocess a cropped plate for VLM / OCR (upscale, CLAHE, sharpen).
Output is single-channel (grayscale) uint8 — plate_reader converts to RGB for Qwen.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np


class PlateEnhancer:
    def __init__(self, scale: int = 3):
        self.scale = scale

    def enhance(self, plate_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if plate_bgr is None or plate_bgr.size == 0:
            return None

        up = cv2.resize(
            plate_bgr,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        kernel = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
            dtype=np.float32,
        )
        sharp = cv2.filter2D(gray, -1, kernel)
        return sharp

    def enhance_plate_for_qwen(self, plate_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Alias for notebooks; same as enhance()."""
        return self.enhance(plate_bgr)

    def variants(self, plate_bgr: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        """Optional: multiple threshold variants for experimentation."""
        if plate_bgr is None or plate_bgr.size == 0:
            return {}

        up = cv2.resize(
            plate_bgr,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        out: Dict[str, np.ndarray] = {"gray": gray.copy()}
        out["clahe"] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out["sharp"] = cv2.filter2D(gray, -1, kernel)
        _, out["otsu"] = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        out["adaptive"] = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        return out


_default = PlateEnhancer(scale=3)


def enhance_plate(plate_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Enhance a BGR plate crop for Qwen (returns grayscale ndarray)."""
    return _default.enhance(plate_bgr)
