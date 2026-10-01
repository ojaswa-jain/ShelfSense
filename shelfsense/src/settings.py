"""Fail clearly on malformed configuration instead of accepting invalid thresholds."""
import json
import math
from pathlib import Path


def load_settings(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read config/default_settings.json.") from error
    bounds = {"blur_warning_threshold": (0, 1000), "detector_confidence": (0.05, 0.95),
              "nms_iou": (0.1, 0.9), "similarity_threshold": (0, 1),
              "ambiguity_margin": (0, 0.5), "minimum_gap_fraction": (0.01, 0.5)}
    if not isinstance(data, dict) or data.get("max_side") not in [800, 1200, 1600, 2048]:
        raise ValueError("max_side must be 800, 1200, 1600 or 2048.")
    for key, (low, high) in bounds.items():
        value = data.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must lie between {low} and {high}.")
        data[key] = float(value)
    return data
