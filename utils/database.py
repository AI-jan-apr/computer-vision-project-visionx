"""
SQLite storage for ground truth keyed by plate number (primary key).

- plate_ground_truth: expected digits/letters/final plate, optional brand & model.
- test_image_map: basename of a file under test_images/ -> which plate_final we expect.

Use db_cli.py to insert rows, then run main.py --compare to check predictions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from utils.color_normalize import colors_match
from utils.config import DEFAULT_DB_PATH


def normalize_plate_final(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip().upper()
    if not s:
        return None
    return " ".join(s.split())


def normalize_token(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip().upper().replace(" ", "")
    return s or None


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | None = None) -> None:
    path = db_path or DEFAULT_DB_PATH
    with _connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS plate_ground_truth (
                plate_final TEXT PRIMARY KEY,
                digits TEXT NOT NULL,
                letters TEXT NOT NULL,
                expected_brand TEXT,
                expected_model TEXT,
                expected_color TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS test_image_map (
                image_basename TEXT PRIMARY KEY,
                plate_final TEXT NOT NULL,
                FOREIGN KEY (plate_final)
                    REFERENCES plate_ground_truth(plate_final)
                    ON DELETE CASCADE
                    ON UPDATE CASCADE
            );
            """
        )
        conn.commit()
        _migrate_plate_table(conn)
        conn.commit()


def _migrate_plate_table(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(plate_ground_truth)").fetchall()}
    if "expected_color" not in cols:
        conn.execute("ALTER TABLE plate_ground_truth ADD COLUMN expected_color TEXT")


def upsert_ground_truth(
    plate_final: str,
    digits: str,
    letters: str,
    *,
    expected_brand: Optional[str] = None,
    expected_model: Optional[str] = None,
    expected_color: Optional[str] = None,
    db_path: Path | None = None,
) -> str:
    """Insert or replace ground truth row. Returns normalized plate_final."""
    key = normalize_plate_final(plate_final)
    if not key:
        raise ValueError("plate_final is empty")
    d = normalize_token(digits) or ""
    L = normalize_token(letters) or ""
    init_db(db_path)
    with _connect(db_path or DEFAULT_DB_PATH) as conn:
        _migrate_plate_table(conn)
        conn.execute(
            """
            INSERT INTO plate_ground_truth
                (plate_final, digits, letters, expected_brand, expected_model, expected_color)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(plate_final) DO UPDATE SET
                digits = excluded.digits,
                letters = excluded.letters,
                expected_brand = excluded.expected_brand,
                expected_model = excluded.expected_model,
                expected_color = excluded.expected_color;
            """,
            (key, d, L, expected_brand, expected_model, expected_color),
        )
        conn.commit()
    return key


def link_test_image(
    image_basename: str,
    plate_final: str,
    *,
    db_path: Path | None = None,
) -> None:
    """Register that this filename (e.g. car01.jpg) should match this plate row."""
    base = Path(image_basename).name
    key = normalize_plate_final(plate_final)
    if not key:
        raise ValueError("plate_final is empty")
    init_db(db_path)
    with _connect(db_path or DEFAULT_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO test_image_map (image_basename, plate_final)
            VALUES (?, ?)
            ON CONFLICT(image_basename) DO UPDATE SET plate_final = excluded.plate_final;
            """,
            (base, key),
        )
        conn.commit()


def get_expected_for_basename(
    image_basename: str,
    *,
    db_path: Path | None = None,
) -> Optional[Dict[str, Any]]:
    """Ground truth dict for a test image filename, or None if not mapped."""
    base = Path(image_basename).name
    init_db(db_path)
    path = db_path or DEFAULT_DB_PATH
    if not path.is_file():
        return None
    with _connect(path) as conn:
        row = conn.execute(
            """
            SELECT g.plate_final, g.digits, g.letters, g.expected_brand, g.expected_model,
                   g.expected_color
            FROM test_image_map m
            JOIN plate_ground_truth g ON g.plate_final = m.plate_final
            WHERE m.image_basename = ?
            """,
            (base,),
        ).fetchone()
    if row is None:
        return None
    return {
        "plate_final": row["plate_final"],
        "digits": row["digits"],
        "letters": row["letters"],
        "expected_brand": row["expected_brand"],
        "expected_model": row["expected_model"],
        "expected_color": row["expected_color"],
    }


def get_ground_truth_by_plate_final(
    plate_final: str,
    *,
    db_path: Path | None = None,
) -> Optional[Dict[str, Any]]:
    """Lookup registry row by normalized plate (after reading the plate from the image)."""
    key = normalize_plate_final(plate_final)
    if not key:
        return None
    init_db(db_path)
    path = db_path or DEFAULT_DB_PATH
    if not path.is_file():
        return None
    with _connect(path) as conn:
        row = conn.execute(
            """
            SELECT plate_final, digits, letters, expected_brand, expected_model, expected_color
            FROM plate_ground_truth
            WHERE plate_final = ?
            """,
            (key,),
        ).fetchone()
    if row is None:
        return None
    return {
        "plate_final": row["plate_final"],
        "digits": row["digits"],
        "letters": row["letters"],
        "expected_brand": row["expected_brand"],
        "expected_model": row["expected_model"],
        "expected_color": row["expected_color"],
    }


def compare_prediction_to_ground_truth(
    expected: Dict[str, Any],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare pipeline summary (brand_model, plate_read) to DB expectations.
    Booleans: plate_final_match, digits_match, letters_match, brand_match, model_match.
    """
    pred_digits = normalize_token(
        (summary.get("plate_read") or {}).get("digits")
        if summary.get("plate_read")
        else None
    )
    pred_letters = normalize_token(
        (summary.get("plate_read") or {}).get("letters")
        if summary.get("plate_read")
        else None
    )
    pred_final = normalize_plate_final(
        (summary.get("plate_read") or {}).get("final")
        if summary.get("plate_read")
        else None
    )

    exp_digits = normalize_token(expected.get("digits"))
    exp_letters = normalize_token(expected.get("letters"))
    exp_final = normalize_plate_final(expected.get("plate_final"))

    brand_info = summary.get("brand_model") or {}
    pred_brand = (brand_info.get("brand") or "").strip() or None
    pred_model = (brand_info.get("model") or "").strip() or None
    exp_brand = (expected.get("expected_brand") or "").strip() or None
    exp_model = (expected.get("expected_model") or "").strip() or None

    def norm_brand(s: Optional[str]) -> Optional[str]:
        return s.strip().upper() if s else None

    pb, eb = norm_brand(pred_brand), norm_brand(exp_brand)
    pm, em = norm_brand(pred_model), norm_brand(exp_model)

    plate_final_match = (
        pred_final is not None and exp_final is not None and pred_final == exp_final
    )
    if exp_digits is None:
        digits_match = None
    elif pred_digits is None:
        digits_match = False
    else:
        digits_match = pred_digits == exp_digits

    if exp_letters is None:
        letters_match = None
    elif pred_letters is None:
        letters_match = False
    else:
        letters_match = pred_letters == exp_letters

    if eb is None:
        brand_match = None
    elif pb is None:
        brand_match = False
    else:
        brand_match = pb == eb

    if em is None:
        model_match = None
    elif pm is None:
        model_match = False
    else:
        model_match = pm == em

    exp_color = (expected.get("expected_color") or "").strip() or None
    vc = summary.get("vehicle_color") or {}
    pred_color_raw = (vc.get("color") or "").strip() or None
    color_match = colors_match(exp_color, pred_color_raw)

    all_plate_ok = (
        plate_final_match
        and (digits_match is True)
        and (letters_match is True)
    )

    failures = []
    if not all_plate_ok:
        failures.append("plate")
    if brand_match is False:
        failures.append("brand")
    if model_match is False:
        failures.append("model")
    if color_match is False:
        failures.append("color")

    registry_consistent = len(failures) == 0

    return {
        "plate_final_match": plate_final_match,
        "digits_match": digits_match,
        "letters_match": letters_match,
        "brand_match": brand_match,
        "model_match": model_match,
        "color_match": color_match,
        "all_plate_fields_match": all_plate_ok,
        "registry_consistent": registry_consistent,
        "registry_mismatch": not registry_consistent,
        "mismatch_fields": failures,
        "expected": expected,
        "predicted": {
            "digits": pred_digits,
            "letters": pred_letters,
            "final": pred_final,
            "brand": pred_brand,
            "model": pred_model,
            "color": pred_color_raw,
        },
    }
