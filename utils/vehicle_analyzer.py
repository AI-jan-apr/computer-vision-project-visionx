from utils.color_predictor import predict_color
from utils.vehicle_vlm_colab import analyze_vehicle_with_colab

def analyze_vehicle(vehicle_crop):
    if vehicle_crop is None:
        return {
            "brand": "Unknown",
            "model": "Unknown",
            "color": "Unknown",
            "color_conf": 0.0,
            "type": "Unknown"
        }

    color_result = predict_color(vehicle_crop)
    vlm_result = analyze_vehicle_with_colab(vehicle_crop)

    return {
        "brand": vlm_result.get("brand", "Unknown"),
        "model": vlm_result.get("model", "Unknown"),
        "color": color_result.get("color", "Unknown"),
        "color_conf": color_result.get("color_conf", 0.0),
        "type": vlm_result.get("type", "Unknown")
    }