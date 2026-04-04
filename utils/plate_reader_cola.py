import cv2
import os
import time
import tempfile
import requests

BASE_URL = "https://blowing-testimony-lock-utilities.trycloudflare.com"

def read_plate(plate_image):
    if plate_image is None:
        return {
            "raw": None,
            "digits": None,
            "letters": None,
            "final": None
        }

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name

    try:
        cv2.imwrite(temp_path, plate_image)

        with open(temp_path, "rb") as f:
            submit_response = requests.post(
                f"{BASE_URL}/submit_plate",
                files={"file": ("plate.png", f, "image/png")},
                timeout=60
            )

        submit_response.raise_for_status()
        submit_data = submit_response.json()
        job_id = submit_data["job_id"]

        for _ in range(120):
            result_response = requests.get(
                f"{BASE_URL}/result/{job_id}",
                timeout=60
            )
            result_response.raise_for_status()
            result_data = result_response.json()

            status = result_data.get("status")

            if status == "done":
                return result_data["result"]

            if status == "error":
                return {
                    "raw": None,
                    "digits": None,
                    "letters": None,
                    "final": None,
                    "error": result_data.get("error")
                }

            time.sleep(2)

        return {
            "raw": None,
            "digits": None,
            "letters": None,
            "final": None,
            "error": "Timed out waiting for result"
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)