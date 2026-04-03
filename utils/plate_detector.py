from ultralytics import YOLO
import cv2


class PlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_plate(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            return None, None

        results = self.model(img)[0]

        if len(results.boxes) == 0:
            return img, None

        best_box = max(results.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())

        plate = img[y1:y2, x1:x2]
        return img, plate

    def detect_plate_with_padding(self, image_path, pad_x_ratio=0.05, pad_y_ratio=0.10):
        img = cv2.imread(image_path)

        if img is None:
            return None, None

        results = self.model(img)[0]

        if len(results.boxes) == 0:
            return img, None

        best_box = max(results.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())

        h, w = img.shape[:2]

        pad_x = int((x2 - x1) * pad_x_ratio)
        pad_y = int((y2 - y1) * pad_y_ratio)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        plate = img[y1:y2, x1:x2]
        return img, plate