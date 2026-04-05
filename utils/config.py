"""Central paths and env overrides for VisionX models."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

# Override with env vars if your Colab export uses different filenames.
VEHICLE_MODEL_PATH = Path(
    os.environ.get(
        "VISIONX_VEHICLE_MODEL",
        str(MODELS_DIR / "car_object_detector" / "car.pt"),
    )
)
PLATE_MODEL_PATH = Path(
    os.environ.get(
        "VISIONX_PLATE_MODEL",
        str(MODELS_DIR / "plate_detector" / "plate.pt"),
    )
)


def _default_brand_weights() -> str:
    d = MODELS_DIR / "car_brand_model_detector"
    for name in ("best.pt", "brand_model.pt", "weights.pt", "last.pt"):
        p = d / name
        if p.is_file():
            return str(p)
    return str(d / "best.pt")


BRAND_MODEL_PATH = Path(
    os.environ.get("VISIONX_BRAND_MODEL", _default_brand_weights())
)

COLOR_MODEL_PATH = Path(
    os.environ.get(
        "VISIONX_COLOR_MODEL",
        str(MODELS_DIR / "car_color_model_augmented.keras"),
    )
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_DB_PATH = Path(
    os.environ.get(
        "VISIONX_DB",
        str(PROJECT_ROOT / "data" / "visionx.db"),
    )
)
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
QWEN_MODEL_ID = os.environ.get(
    "VISIONX_QWEN_MODEL",
    "Qwen/Qwen2.5-VL-3B-Instruct",
)
# Plate OCR needs only a few dozen tokens; lower = faster decode (still safe for DIGITS/LETTERS/FINAL).
QWEN_MAX_NEW_TOKENS = int(os.environ.get("VISIONX_QWEN_MAX_NEW_TOKENS", "64"))
# Requires: pip install flash-attn (CUDA). Falls back to default attention if load fails.
QWEN_FLASH_ATTN2 = os.environ.get("VISIONX_QWEN_FLASH_ATTN2", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
