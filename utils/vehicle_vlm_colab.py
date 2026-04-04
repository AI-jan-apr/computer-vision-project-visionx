import cv2
import os
import time
import tempfile
import requests

BASE_URL = "https://draws-shorter-meaningful-gregory.trycloudflare.com"

def analyze_vehicle_with_colab(vehicle_crop):
    if vehicle_crop is None:
        return {
            "brand": "Unknown",
            "model": "Unknown",
            "type": "Unknown"
        }

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name

    try:
        cv2.imwrite(temp_path, vehicle_crop)

        with open(temp_path, "rb") as f:
            submit_response = requests.post(
                f"{BASE_URL}/submit_vehicle",
                files={"file": ("vehicle.png", f, "image/png")},
                timeout=60
            )

        submit_response.raise_for_status()
        submit_data = submit_response.json()
        job_id = submit_data["job_id"]

        for _ in range(120):
            result_response = requests.get(
                f"{BASE_URL}/result_vehicle/{job_id}",
                timeout=60
            )
            result_response.raise_for_status()
            result_data = result_response.json()

            status = result_data.get("status")

            if status == "done":
                return result_data["result"]

            if status == "error":
                return {
                    "brand": "Unknown",
                    "model": "Unknown",
                    "type": "Unknown",
                    "error": result_data.get("error")
                }

            time.sleep(2)

        return {
            "brand": "Unknown",
            "model": "Unknown",
            "type": "Unknown",
            "error": "Timed out waiting for vehicle VLM result"
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)