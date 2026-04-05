"""
VisionX end-to-end demo: vehicle box → brand/model → plate box → enhance → Qwen plate read.

Place weights under:
  models/car_object_detector/car.pt
  models/plate_detector/plate.pt
  models/car_brand_model_detector/best.pt

Usage:
  python main.py --image path/to/car.jpg
  python main.py --image path/to/car.jpg --skip-qwen
  python main.py --image path/to/car.jpg --skip-brand
  python main.py --image test_images/car01.jpg --compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from utils.car_brand_model_detector import predict_brand_model
from utils.color_predictor import predict_color
from utils.config import DEFAULT_DB_PATH, DEFAULT_OUTPUT_DIR, PROJECT_ROOT, TEST_IMAGES_DIR
from utils.database import (
    compare_prediction_to_ground_truth,
    get_expected_for_basename,
    get_ground_truth_by_plate_final,
)
from utils.draw import draw_bbox
from utils.plate_detector import detect_plate, plate_bbox_in_full_image
from utils.plate_enhancer import enhance_plate
from utils.plate_reader import read_plate
from utils.vehicle_detector import detect_vehicle


def _print_section(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def _log(quiet: bool, *args, **kwargs) -> None:
    if not quiet:
        print(*args, **kwargs)


def attach_db_comparison_to_summary(
    summary: dict,
    image_basename: str | None,
    *,
    db_path: Path | None = None,
) -> dict | None:
    """
    Fill summary[\"db_comparison\"] from registry (by image name, else by read plate).
    Returns the comparison dict or None.
    """
    path = db_path or DEFAULT_DB_PATH
    expected = None
    if image_basename:
        expected = get_expected_for_basename(image_basename, db_path=path)
    if expected is None:
        pr = summary.get("plate_read") or {}
        pf = pr.get("final")
        if pf:
            expected = get_ground_truth_by_plate_final(pf, db_path=path)
    if expected is None:
        return None
    comp = compare_prediction_to_ground_truth(expected, summary)
    summary["db_comparison"] = comp
    return comp


def run_pipeline(
    image_path: Path,
    out_dir: Path,
    *,
    skip_brand: bool = False,
    skip_color: bool = False,
    skip_qwen: bool = False,
    quiet: bool = False,
) -> dict:
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    summary: dict = {"image": str(image_path)}

    if not quiet:
        _print_section("VEHICLE")
    vehicle = None
    try:
        vehicle = detect_vehicle(bgr)
    except FileNotFoundError as e:
        _log(quiet, e)

    summary["vehicle"] = None
    if vehicle:
        summary["vehicle"] = {
            "bbox": vehicle["bbox"],
            "confidence": round(vehicle["confidence"], 4),
        }
        cv2.imwrite(str(out_dir / "01_vehicle_annotated.jpg"), vehicle["annotated"])
        cv2.imwrite(str(out_dir / "02_vehicle_crop.jpg"), vehicle["crop"])
        _log(quiet, "bbox:", vehicle["bbox"], "conf:", vehicle["confidence"])
    else:
        _log(quiet, "No vehicle detected.")

    if not quiet:
        _print_section("BRAND / MODEL")
    brand_info = None
    if skip_brand:
        _log(quiet, "Skipped (--skip-brand).")
    elif vehicle and vehicle.get("crop") is not None:
        try:
            brand_info = predict_brand_model(vehicle["crop"])
            summary["brand_model"] = brand_info
            _log(quiet, json.dumps(brand_info, indent=2))
        except FileNotFoundError as e:
            _log(quiet, e)
            summary["brand_model"] = None
    else:
        _log(quiet, "No vehicle crop; skipped.")
        summary["brand_model"] = None

    if not quiet:
        _print_section("COLOR")
    if skip_color:
        _log(quiet, "Skipped (skip_color).")
        summary["vehicle_color"] = None
    elif vehicle and vehicle.get("crop") is not None:
        try:
            col = predict_color(vehicle["crop"])
            summary["vehicle_color"] = col
            _log(quiet, json.dumps(col, indent=2))
        except FileNotFoundError as e:
            _log(quiet, e)
            summary["vehicle_color"] = None
        except Exception as e:
            _log(quiet, "Color model error:", e)
            summary["vehicle_color"] = None
    else:
        _log(quiet, "No vehicle crop; skipped.")
        summary["vehicle_color"] = None

    if not quiet:
        _print_section("PLATE")
    plate_roi = vehicle["crop"] if vehicle else bgr
    plate = None
    try:
        plate = detect_plate(plate_roi)
    except FileNotFoundError as e:
        _log(quiet, e)

    summary["plate"] = None
    if plate:
        summary["plate"] = {
            "bbox_in_roi": plate["bbox"],
            "confidence": round(plate["confidence"], 4),
        }
        if vehicle:
            summary["plate"]["bbox_in_full_image"] = plate_bbox_in_full_image(
                vehicle["bbox"], plate["bbox"]
            )
        cv2.imwrite(str(out_dir / "03_plate_on_roi.jpg"), plate["annotated"])
        cv2.imwrite(str(out_dir / "04_plate_crop.jpg"), plate["crop"])
        _log(quiet, "bbox (in ROI):", plate["bbox"], "conf:", plate["confidence"])
    else:
        _log(quiet, "No plate detected in ROI.")

    if not quiet:
        _print_section("ENHANCE + READ")
    plate_read = None
    if plate is None:
        _log(quiet, "No plate crop; skipped.")
        summary["plate_read"] = None
    else:
        enhanced = enhance_plate(plate["crop"])
        if enhanced is None:
            _log(quiet, "Enhancement failed.")
            summary["plate_read"] = None
        else:
            cv2.imwrite(str(out_dir / "05_plate_enhanced.png"), enhanced)
            if skip_qwen:
                _log(quiet, "Skipped Qwen (--skip-qwen). Enhanced image saved.")
                summary["plate_read"] = None
            else:
                plate_read = read_plate(enhanced)
                summary["plate_read"] = plate_read
                _log(quiet, "raw:", plate_read.get("raw"))
                _log(
                    quiet,
                    "digits:",
                    plate_read.get("digits"),
                    "letters:",
                    plate_read.get("letters"),
                )
                _log(quiet, "final:", plate_read.get("final"))

    if not quiet:
        _print_section("VISUALIZATION (full image)")
    vis = bgr.copy()
    if vehicle:
        vis = draw_bbox(
            vis,
            vehicle["bbox"],
            label=f"vehicle {vehicle['confidence']:.2f}",
            color=(0, 200, 0),
        )
    if plate and vehicle:
        g = plate_bbox_in_full_image(vehicle["bbox"], plate["bbox"])
        vis = draw_bbox(
            vis,
            g,
            label=f"plate {plate['confidence']:.2f}",
            color=(0, 165, 255),
        )
    elif plate and not vehicle:
        vis = draw_bbox(
            vis,
            plate["bbox"],
            label=f"plate {plate['confidence']:.2f}",
            color=(0, 165, 255),
        )
    cv2.imwrite(str(out_dir / "00_overview.jpg"), vis)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def _print_comparison(comp: dict) -> None:
    _print_section("DB COMPARISON")
    print("plate_final_match:", comp["plate_final_match"])
    print("digits_match:", comp["digits_match"])
    print("letters_match:", comp["letters_match"])
    print("brand_match:", comp["brand_match"])
    print("model_match:", comp["model_match"])
    print("color_match:", comp.get("color_match"))
    print("all_plate_fields_match:", comp["all_plate_fields_match"])
    print("registry_consistent:", comp.get("registry_consistent"))
    print("mismatch_fields:", comp.get("mismatch_fields"))
    print("expected:", json.dumps(comp["expected"], ensure_ascii=False))
    print("predicted:", json.dumps(comp["predicted"], ensure_ascii=False))


def main() -> None:
    candidates = sorted(TEST_IMAGES_DIR.glob("*.jpg")) + sorted(
        TEST_IMAGES_DIR.glob("*.jpeg")
    )
    default_path = candidates[0] if candidates else None

    p = argparse.ArgumentParser(description="VisionX vehicle + plate + Qwen pipeline")
    p.add_argument(
        "--image",
        type=Path,
        default=default_path,
        help="Path to input image (default: first jpg/jpeg in test_images/)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    p.add_argument("--skip-brand", action="store_true", help="Do not run brand/model classifier")
    p.add_argument(
        "--skip-color",
        action="store_true",
        help="Do not run Keras vehicle color model",
    )
    p.add_argument(
        "--skip-qwen",
        action="store_true",
        help="Skip Qwen2.5-VL plate read (faster; no plate text → DB match by plate number won't work)",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite path for ground truth (default: data/visionx.db)",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Compare pipeline output to DB using image basename -> expected plate row",
    )
    args = p.parse_args()

    if args.image is None:
        print("No --image given and test_images/ is empty.", file=sys.stderr)
        sys.exit(1)

    try:
        summary = run_pipeline(
            args.image,
            args.out,
            skip_brand=args.skip_brand,
            skip_color=args.skip_color,
            skip_qwen=args.skip_qwen,
        )
        if args.compare:
            comp = attach_db_comparison_to_summary(
                summary, args.image.name, db_path=args.db
            )
            if comp is None:
                print(
                    f"\n[compare] No DB row for image {args.image.name!r} or plate read. "
                    f"Import Excel: python db_cli.py import-excel \"path.xlsx\" "
                    f"or link: python db_cli.py link --image ...",
                    file=sys.stderr,
                )
            else:
                _print_comparison(comp)
                with open(args.out / "summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nDone. Outputs in: {args.out.resolve()}")
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
