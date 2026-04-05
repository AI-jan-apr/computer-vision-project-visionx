"""Draw bounding boxes on BGR images (OpenCV)."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

Color = Tuple[int, int, int]


def draw_bbox(
    image: np.ndarray,
    bbox: List[int],
    label: str = "",
    color: Color = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of `image` with rectangle (x1,y1,x2,y2) and optional label."""
    out = image.copy()
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out
