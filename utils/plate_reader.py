"""
Read Saudi plate text with Qwen2.5-VL (local transformers).
Expects a plate crop (BGR or grayscale ndarray); returns structured digits/letters/final.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

import cv2
import numpy as np
from PIL import Image

from utils.config import (
    QWEN_FLASH_ATTN2,
    QWEN_MAX_NEW_TOKENS,
    QWEN_MODEL_ID,
)

_reader_instance: Optional["PlateReader"] = None

SAUDI_PLATE_PROMPT = """
You are reading a Saudi vehicle license plate image.

Important:
- Read only the BOTTOM ENGLISH part of the Saudi plate.
- The bottom-left side contains exactly 4 English digits.
- The bottom-right side contains exactly 3 English letters.
- Ignore the Arabic text.
- Do not explain.
- Return only in this exact format:

DIGITS: XXXX
LETTERS: XXX
FINAL: XXXX XXX
""".strip()


def _numpy_to_pil(plate: np.ndarray) -> Image.Image:
    if plate.ndim == 2:
        rgb = cv2.cvtColor(plate, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def parse_plate_output(text: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if not text:
        return None, None, None
    digits_m = re.search(r"DIGITS:\s*([A-Z0-9?]+)", text, re.IGNORECASE)
    letters_m = re.search(r"LETTERS:\s*([A-Z0-9?]+)", text, re.IGNORECASE)
    final_m = re.search(r"FINAL:\s*([A-Z0-9? ]+)", text, re.IGNORECASE)
    digits = digits_m.group(1).strip().upper() if digits_m else None
    letters = letters_m.group(1).strip().upper() if letters_m else None
    final = final_m.group(1).strip().upper() if final_m else None
    return digits, letters, final


class PlateReader:
    """Loads Qwen2.5-VL on first use (heavy)."""

    def __init__(self, model_id: Optional[str] = None) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        t_load = time.perf_counter()
        self._torch = torch
        self.model_id = model_id or QWEN_MODEL_ID
        self.max_new_tokens = max(16, QWEN_MAX_NEW_TOKENS)
        device_hint = "CUDA" if torch.cuda.is_available() else "CPU"
        logger.info(
            "VisionX Qwen: بدء تحميل الموديل %s على %s (أول مرة قد تستغرق وقتاً طويلاً)…",
            self.model_id,
            device_hint,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        load_kw: dict = {"torch_dtype": dtype}
        if QWEN_FLASH_ATTN2 and torch.cuda.is_available():
            load_kw["attn_implementation"] = "flash_attention_2"
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **load_kw,
            )
        except Exception:
            if "attn_implementation" in load_kw:
                load_kw.pop("attn_implementation", None)
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    **load_kw,
                )
            else:
                raise
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.to("cpu")
        self.model.eval()
        dev = next(self.model.parameters()).device
        logger.info(
            "VisionX Qwen: الموديل جاهز على %s بعد %.1f ث (القراءة التالية أسرع).",
            dev,
            time.perf_counter() - t_load,
        )

    def _generate(self, pil_image: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image.convert("RGB")},
                    {"type": "text", "text": SAUDI_PLATE_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[pil_image.convert("RGB")],
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        t_inf = time.perf_counter()
        logger.info("VisionX Qwen: تشغيل قراءة اللوحة (inference)…")
        with self._torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        logger.info(
            "VisionX Qwen: انتهت قراءة اللوحة بعد %.1f ث.",
            time.perf_counter() - t_inf,
        )

        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def read_raw_from_array(self, plate_image: np.ndarray) -> str:
        pil = _numpy_to_pil(plate_image)
        return self._generate(pil)

    def read_and_parse_array(self, plate_image: np.ndarray) -> Dict[str, Any]:
        raw = self.read_raw_from_array(plate_image)
        d, l, f = parse_plate_output(raw)
        return {"raw": raw, "digits": d, "letters": l, "final": f}

    def read_and_parse_path(self, image_path: str) -> Dict[str, Any]:
        pil = Image.open(image_path).convert("RGB")
        raw = self._generate(pil)
        d, l, f = parse_plate_output(raw)
        return {"raw": raw, "digits": d, "letters": l, "final": f}


def get_plate_reader() -> PlateReader:
    global _reader_instance
    if _reader_instance is None:
        _reader_instance = PlateReader()
    return _reader_instance


def is_plate_reader_loaded() -> bool:
    return _reader_instance is not None


def read_plate(plate_image: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    Read plate from a numpy crop (BGR or grayscale).
    Loads Qwen on first call.
    """
    if plate_image is None or plate_image.size == 0:
        return {"raw": None, "digits": None, "letters": None, "final": None}
    return get_plate_reader().read_and_parse_array(plate_image)
