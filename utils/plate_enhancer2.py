import cv2
import numpy as np


class PlateEnhancer:
    def __init__(self, scale=3):
        self.scale = scale

    def enhance_plate_for_qwen(self, plate):
        if plate is None:
            return None

        up = cv2.resize(
            plate,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        kernel = np.array([
            [0, -1,  0],
            [-1, 5, -1],
            [0, -1,  0]
        ])
        sharp = cv2.filter2D(gray, -1, kernel)

        return sharp

    def build_variants(self, plate):
        if plate is None:
            return {}

        up = cv2.resize(
            plate,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

        variants = {}

        variants["gray"] = gray

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        variants["clahe"] = clahe

        sharp_kernel = np.array([
            [0, -1,  0],
            [-1, 5, -1],
            [0, -1,  0]
        ])
        sharp = cv2.filter2D(gray, -1, sharp_kernel)
        variants["sharp"] = sharp

        _, otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        variants["otsu"] = otsu

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )
        variants["adaptive"] = adaptive

        return variants


# =========================
# Simple helper functions
# =========================

_enhancer = PlateEnhancer(scale=3)

def enhance_plate(plate):
    return _enhancer.enhance_plate_for_qwen(plate)

def get_plate_variants(plate):
    return _enhancer.build_variants(plate)