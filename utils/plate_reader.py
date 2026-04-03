import re
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


class PlateReader:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model_id = model_id

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def read_plate(self, image_path):
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
                        "text": """
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

    def parse_plate_output(self, text):
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