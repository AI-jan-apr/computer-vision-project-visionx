# pip install gradio_client
from utils.vehicle_detector import detect_vehicle
from utils.plate_detector2 import detect_plate
from utils.plate_enhancer2 import enhance_plate
from utils.plate_reader_cola import read_plate
import cv2
import os

def main():
    image_path = "test_images/WhatsApp Image 2026-04-04 at 22.44.38.jpeg"
    os.makedirs("outputs", exist_ok=True)

    # Vehicle detection
    vehicle_crop, vehicle_bbox, vehicle_conf = detect_vehicle(image_path)
    print("\n===== VEHICLE DETECTION =====")
    print("Vehicle BBox:", vehicle_bbox)
    print("Vehicle Confidence:", vehicle_conf)

    if vehicle_crop is not None:
        cv2.imwrite("outputs/vehicle_crop.jpg", vehicle_crop)
        print("Saved: outputs/vehicle_crop.jpg")
    else:
        print("No vehicle detected")

    # Plate detection
    plate_crop, plate_bbox, plate_conf = detect_plate(image_path)
    print("\n===== PLATE DETECTION =====")
    print("Plate BBox:", plate_bbox)
    print("Plate Confidence:", plate_conf)

    if plate_crop is not None:
        cv2.imwrite("outputs/plate_crop.jpg", plate_crop)
        print("Saved: outputs/plate_crop.jpg")

        # Enhance plate
        enhanced_plate = enhance_plate(plate_crop)
        if enhanced_plate is not None:
            cv2.imwrite("outputs/plate_enhanced.jpg", enhanced_plate)
            print("Saved: outputs/plate_enhanced.jpg")

            # Read plate
            plate_result = read_plate(enhanced_plate)

            print("\n===== PLATE READING =====")
            print("Raw Output:", plate_result["raw"])
            print("Digits:", plate_result["digits"])
            print("Letters:", plate_result["letters"])
            print("Final:", plate_result["final"])
        else:
            print("Plate enhancement failed")
    else:
        print("No plate detected")


if __name__ == "__main__":
    main()