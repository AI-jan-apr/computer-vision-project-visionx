"""
Map free-text colors (Excel / user) to the same buckets as color_predictor.CLASS_NAMES:
black, blue, red, white, silver, grey
"""

from __future__ import annotations

import re
from typing import Optional

MODEL_COLOR_LABELS = frozenset({"black", "blue", "red", "white", "silver", "grey"})


def color_text_to_bucket(text: Optional[str]) -> Optional[str]:
    """Return a model bucket name or None if unknown."""
    if text is None:
        return None
    t = re.sub(r"[_\-]+", " ", str(text).strip().lower())
    if not t:
        return None

    if t in MODEL_COLOR_LABELS:
        return t

    # Order matters: more specific phrases first
    if any(x in t for x in ("navy", "matte blue")) or (t == "blue" or t.endswith(" blue")):
        return "blue"
    if "red" in t or "maroon" in t:
        return "red"
    if "white" in t or "pearl" in t or "ivory" in t:
        return "white"
    if any(x in t for x in ("black", "onyx", "carbon", "obsidian")):
        return "black"
    if "silver" in t or "titanium" in t or "chrome" in t:
        return "silver"
    if any(x in t for x in ("grey", "gray", "graphite", "selenite", "bronze")):
        return "grey"
    if "green" in t or "sarge" in t:
        return "grey"

    return None


def colors_match(expected_label: Optional[str], predicted_label: Optional[str]) -> Optional[bool]:
    """None if either side missing bucket; True/False if both mapped."""
    eb = color_text_to_bucket(expected_label)
    pl = (predicted_label or "").strip().lower() if predicted_label else None
    pb = pl if pl in MODEL_COLOR_LABELS else color_text_to_bucket(predicted_label)
    if eb is None or pb is None:
        return None
    return eb == pb
