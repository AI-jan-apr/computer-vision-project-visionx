"""Import ground-truth rows from Excel (License Plate, Brand, Model, Color)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

from utils.database import init_db, upsert_ground_truth

# Flexible header matching (English / Arabic sheet names OK)
PLATE_ALIASES = frozenset(
    {"license plate", "plate", "platenumber", "plate_number", "رقم اللوحة", "لوحة"}
)
BRAND_ALIASES = frozenset({"brand", "make", "الشركة", "الماركة"})
MODEL_ALIASES = frozenset({"model", "الموديل"})
COLOR_ALIASES = frozenset({"color", "colour", "اللون"})


def _norm_col(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _find_column(columns: List[str], aliases: frozenset) -> Optional[str]:
    for c in columns:
        n = _norm_col(c)
        if n in aliases or any(a in n for a in aliases if len(a) > 3):
            return c
        for a in aliases:
            if n == a or n.replace(" ", "") == a.replace(" ", ""):
                return c
    return None


def parse_license_plate_cell(value: Any) -> Tuple[str, str, str]:
    """
    From '6304 LAJ' -> (plate_final, digits, letters).
    """
    s = str(value).strip().upper()
    s = " ".join(s.split())
    if not s:
        raise ValueError("empty license plate")

    parts = s.split()
    if len(parts) >= 2:
        digits = "".join(c for c in parts[0] if c.isdigit())
        letters = "".join(c for c in parts[-1] if c.isalpha())
        if digits and letters:
            return f"{digits} {letters}", digits, letters

    m = re.match(r"^(\d{1,4})\s*([A-Z]{1,3})$", s.replace(" ", ""))
    if m:
        d, L = m.group(1), m.group(2)
        return f"{d} {L}", d, L

    raise ValueError(f"unrecognized plate format: {value!r}")


def import_ground_truth_xlsx(
    xlsx_path: str | Path,
    *,
    sheet_name: str | int = 0,
    db_path: Optional[Path] = None,
) -> int:
    """
    Read Excel and upsert each row into plate_ground_truth.
    Returns number of rows imported.
    """
    import pandas as pd

    path = Path(xlsx_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    df = pd.read_excel(path, sheet_name=sheet_name)
    cols = list(df.columns)
    c_plate = _find_column(cols, PLATE_ALIASES) or cols[0]
    c_brand = _find_column(cols, BRAND_ALIASES)
    c_model = _find_column(cols, MODEL_ALIASES)
    c_color = _find_column(cols, COLOR_ALIASES)

    if c_plate is None:
        raise ValueError(
            f"Could not find License Plate column. Columns: {cols}"
        )

    init_db(db_path)
    n = 0
    for _, row in df.iterrows():
        raw_plate = row.get(c_plate)
        if raw_plate is None or (isinstance(raw_plate, float) and str(raw_plate) == "nan"):
            continue
        try:
            final, digits, letters = parse_license_plate_cell(raw_plate)
        except ValueError:
            continue
        brand = row.get(c_brand) if c_brand else None
        model = row.get(c_model) if c_model else None
        color = row.get(c_color) if c_color else None
        if brand is not None and str(brand).lower() == "nan":
            brand = None
        if model is not None and str(model).lower() == "nan":
            model = None
        if color is not None and str(color).lower() == "nan":
            color = None
        upsert_ground_truth(
            final,
            digits,
            letters,
            expected_brand=str(brand).strip() if brand is not None else None,
            expected_model=str(model).strip() if model is not None else None,
            expected_color=str(color).strip() if color is not None else None,
            db_path=db_path,
        )
        n += 1
    return n
