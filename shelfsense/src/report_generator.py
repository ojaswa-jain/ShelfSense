"""Full evidence JSON and typed, long-format CSV without dropping report fields."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


DISCLAIMER = ("Visible facings are not total inventory. Occupancy is box-based visual coverage. "
              "Potential gaps, lower counts and display-facing shortfalls require manual verification. "
              "No hidden stock, sales or purchase quantities are inferred.")


def make_report(image_id: str, detections: list[dict], original: list[dict],
                corrections: list[dict], regions: list[dict], analysis: dict,
                settings: dict, comparison: dict | None = None,
                coordinate_info: dict | None = None, catalogue: list[dict] | None = None,
                timestamp: str | None = None) -> dict:
    automatic = any(d.get("source") == "model" for d in original)
    # Also preserve automatic mode when a model legitimately returns no boxes.
    automatic = automatic or settings.get("mode") == "automatic" or settings.get("initial_detection_mode") == "automatic"
    mode = "mixed" if automatic and corrections else "automatic" if automatic else "manual"
    payload = {"schema_version": "1.0", "audit_timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
               "image_identifier": image_id, "analysis_mode": mode, "disclaimer": DISCLAIMER,
               "coordinate_info": coordinate_info or {}, "regions": regions,
               "original_model_result": original if automatic else [],
               "initial_analysis_result": original, "corrected_result": detections,
               "user_corrections": corrections, "settings": settings, "analysis": analysis,
               "catalogue_snapshot": catalogue or [],
               "reference_comparison": comparison or {"status": "Not performed"}}
    return deepcopy(payload)


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)


def to_csv(report: dict) -> str:
    """One typed row per section item; nested evidence remains valid JSON."""
    rows = []
    for section, value in report.items():
        items = value if isinstance(value, list) else [value]
        if not items:
            items = [[]]
        for index, item in enumerate(items):
            rows.append({"section": section, "index": index,
                         "value_json": json.dumps(item, ensure_ascii=False, allow_nan=False)})
    return pd.DataFrame(rows, columns=["section", "index", "value_json"]).to_csv(index=False)


def evidence_hash(report: dict) -> str:
    return hashlib.sha256(to_json(report).encode()).hexdigest()
