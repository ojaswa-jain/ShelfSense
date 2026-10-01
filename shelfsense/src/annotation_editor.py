"""Validation and non-destructive corrections, independent of Streamlit."""
from copy import deepcopy
import math
from uuid import uuid4


def validate_box(box: list[float], width: int, height: int) -> list[float]:
    if len(box) != 4:
        raise ValueError("A box needs x1, y1, x2, y2.")
    values = [float(value) for value in box]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Box coordinates must be finite numbers.")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Box must have positive area inside {width} × {height}.")
    return values


def validate_regions(rows: list[dict], width: int, height: int) -> list[dict]:
    if not rows:
        raise ValueError("Define at least one shelf region.")
    result, names = [], set()
    for row in rows:
        name = str(row.get("name") or "").strip()
        if not name or name in names:
            raise ValueError("Shelf region names must be non-empty and unique.")
        box = validate_box([row[key] for key in ("x1", "y1", "x2", "y2")], width, height)
        for other in result:
            a, b, c, d = other["box"]
            if min(c, box[2]) > max(a, box[0]) and min(d, box[3]) > max(b, box[1]):
                raise ValueError("Shelf regions must not overlap.")
        result.append({"name": name, "box": box})
        names.add(name)
    return result


def manual_detection(box: list[float], product_id: str | None = None) -> dict:
    return {"id": uuid4().hex[:12], "box": list(box), "confidence": None,
            "source": "manual", "product_id": product_id,
            "status": "Manual identity" if product_id else "Unknown product",
            "similarity": None, "candidates": []}


def editor_rows(detections: list[dict]) -> list[dict]:
    return [{"id": d["id"], "x1": d["box"][0], "y1": d["box"][1],
             "x2": d["box"][2], "y2": d["box"][3],
             "product_id": d.get("product_id") or "", "remove": False}
            for d in detections]


def apply_corrections(baseline: list[dict], rows: list[dict], width: int,
                      height: int, product_ids: set[str]) -> tuple[list[dict], list[dict]]:
    """Rebuild from an editable snapshot; keep immutable input and an event diff."""
    before = {d["id"]: d for d in baseline}
    result, events, seen = [], [], set()
    for row in rows:
        identity = row.get("id")
        identity = str(identity) if identity else uuid4().hex[:12]
        if identity in seen:
            raise ValueError("Duplicate box IDs are not allowed.")
        seen.add(identity)
        if row.get("remove", False):
            continue
        box = validate_box([row[key] for key in ("x1", "y1", "x2", "y2")], width, height)
        pid = str(row.get("product_id") or "").strip() or None
        if pid and pid not in product_ids:
            raise ValueError(f"Unknown catalogue ID: {pid}")
        old = before.get(identity)
        detection = deepcopy(old) if old else manual_detection(box, pid)
        detection["id"] = identity
        changed_box = old is not None and box != old["box"]
        changed_pid = old is not None and pid != old.get("product_id")
        detection["box"] = box
        if changed_box or changed_pid or old is None:
            detection.update(product_id=pid, similarity=None, candidates=[],
                             status="Manual identity" if pid else "Unknown product")
            if changed_box:
                detection.update(confidence=None, source="corrected")
            events.append({"action": "edit" if old else "add", "id": identity,
                           "before": deepcopy(old), "after": deepcopy(detection)})
        result.append(detection)
    remaining = {d["id"] for d in result}
    for identity, old in before.items():
        if identity not in remaining:
            events.append({"action": "remove", "id": identity, "before": deepcopy(old)})
    return result, events
