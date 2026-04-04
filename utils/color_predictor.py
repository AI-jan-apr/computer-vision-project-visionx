import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/car_color_model_augmented.keras"

# عدلي ترتيب الألوان إذا كان عندك ترتيب مختلف وقت التدريب
CLASS_NAMES =['black', 'blue', 'red', 'white', 'silver', 'grey']


IMG_SIZE = (224, 224)

_color_model = None


def load_color_model():
    global _color_model
    if _color_model is None:
        _color_model = tf.keras.models.load_model(MODEL_PATH)
    return _color_model


def preprocess_color_image(vehicle_crop):
    if vehicle_crop is None:
        return None

    # BGR -> RGB
    img = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    # غالبًا تدريب Keras كان على 0..1
    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)
    return img


def predict_color(vehicle_crop):
    model = load_color_model()
    x = preprocess_color_image(vehicle_crop)

    if x is None:
        return {
            "color": "Unknown",
            "color_conf": 0.0
        }

    preds = model.predict(x, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    best_conf = float(preds[best_idx])

    if best_idx >= len(CLASS_NAMES):
        return {
            "color": "Unknown",
            "color_conf": best_conf
        }

    return {
        "color": CLASS_NAMES[best_idx],
        "color_conf": round(best_conf, 4)
    }