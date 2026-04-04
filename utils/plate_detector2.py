from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "models/plate_detector/best.pt"

plate_model = YOLO(MODEL_PATH)


def detect_plate(image_path):
 

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = plate_model(image)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None, None, None

    best_idx = boxes.conf.argmax().item()

    box = boxes.xyxy[best_idx].tolist()
    conf = float(boxes.conf[best_idx].item())

    x1, y1, x2, y2 = map(int, box)

    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    plate_crop = image[y1:y2, x1:x2]
    bbox = [x1, y1, x2, y2]

    return plate_crop, bbox, conf