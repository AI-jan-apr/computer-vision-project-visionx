import re
import os
import cv2
import torch
import tempfile
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


class PlateReader:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model_id = model_id

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32
      )
        self.model.to(device)
        self.model.eval()

    def _build_prompt(self):
        return """
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
"""

    def _generate_from_image_path(self, image_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path
                    },
                    {
                        "type": "text",
                        "text": self._build_prompt()
                    }
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs = [Image.open(image_path).convert("RGB")]

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=128
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text

    def read_plate(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Plate image not found: {image_path}")

        return self._generate_from_image_path(image_path)

    def read_plate_from_array(self, plate_image):
        if plate_image is None:
            return None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name

        try:
            if len(plate_image.shape) == 2:
                cv2.imwrite(temp_path, plate_image)
            else:
                cv2.imwrite(temp_path, plate_image)

            return self._generate_from_image_path(temp_path)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def parse_plate_output(self, text):
        if not text:
            return None, None, None

        digits_match = re.search(r"DIGITS:\s*([A-Z0-9?]+)", text, re.IGNORECASE)
        letters_match = re.search(r"LETTERS:\s*([A-Z0-9?]+)", text, re.IGNORECASE)
        final_match = re.search(r"FINAL:\s*([A-Z0-9? ]+)", text, re.IGNORECASE)

        digits = digits_match.group(1).strip().upper() if digits_match else None
        letters = letters_match.group(1).strip().upper() if letters_match else None
        final = final_match.group(1).strip().upper() if final_match else None

        return digits, letters, final

    def read_and_parse(self, image_path):
        raw_output = self.read_plate(image_path)
        digits, letters, final = self.parse_plate_output(raw_output)

        return {
            "raw": raw_output,
            "digits": digits,
            "letters": letters,
            "final": final
        }

    def read_and_parse_from_array(self, plate_image):
        raw_output = self.read_plate_from_array(plate_image)
        digits, letters, final = self.parse_plate_output(raw_output)

        return {
            "raw": raw_output,
            "digits": digits,
            "letters": letters,
            "final": final
        }


_reader = None

def get_plate_reader():
    global _reader
    if _reader is None:
        _reader = PlateReader()
    return _reader

def read_plate(plate_image):
    reader = get_plate_reader()
    result = reader.read_and_parse_from_array(plate_image)
    return result