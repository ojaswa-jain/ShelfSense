"""Measure precision/recall, facing-count MAE and inference time on labelled images.

Evaluation requires mirrored images and labels directories in YOLO format. Every
image must have a .txt label file (empty for negative images). One generic class 0.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.image_preprocessing import load_rgb
from src.product_detector import ProductDetector
from src.shelf_analyzer import intersection


def iou(a: list[float], b: list[float]) -> float:
    cut = intersection(a, b)
    if not cut:
        return 0.0
    shared = (cut[2] - cut[0]) * (cut[3] - cut[1])
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return shared / (area(a) + area(b) - shared)


def score_boxes(detections: list[dict], truth: list[list[float]], threshold: float) -> tuple[int, int, int]:
    remaining = set(range(len(truth)))
    tp = 0
    for detection in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        candidates = [(iou(detection["box"], truth[i]), i) for i in remaining]
        if candidates:
            overlap, index = max(candidates)
            if overlap >= threshold:
                tp += 1
                remaining.remove(index)
    return tp, len(detections) - tp, len(truth) - tp


def read_labels(path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        values = [float(v) for v in line.split()]
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO label: {path}")
        cls, cx, cy, bw, bh = values
        if cls != 0 or not all(0 <= v <= 1 for v in [cx, cy, bw, bh]) or min(bw, bh) <= 0:
            raise ValueError(f"Expected normalized class-0 boxes: {path}")
        box = [(cx - bw / 2) * width, (cy - bh / 2) * height,
               (cx + bw / 2) * width, (cy + bh / 2) * height]
        if min(box[:2]) < -1e-3 or box[2] > width + 1e-3 or box[3] > height + 1e-3:
            raise ValueError(f"Out-of-image label: {path}")
        boxes.append(box)
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/detector_evaluation.json"))
    args = parser.parse_args()
    if not 0 < args.iou <= 1 or not 0 <= args.confidence <= 1:
        parser.error("Invalid thresholds.")
    paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        parser.error("No evaluation images found.")
    model = ProductDetector(args.weights, args.device)
    rows, total_tp, total_fp, total_fn = [], 0, 0, 0
    for path in paths:
        label = args.labels / path.relative_to(args.images).with_suffix(".txt")
        if not label.is_file():
            raise ValueError(f"Missing ground truth; refusing to treat it as empty: {label}")
        rgb = load_rgb(path)
        truth = read_labels(label, rgb.shape[1], rgb.shape[0])
        if not rows:
            model.detect(rgb, args.confidence)  # Exclude one warm-up inference.
        start = time.perf_counter()
        detections = model.detect(rgb, args.confidence)
        seconds = time.perf_counter() - start
        tp, fp, fn = score_boxes(detections, truth, args.iou)
        total_tp, total_fp, total_fn = total_tp + tp, total_fp + fp, total_fn + fn
        rows.append({"image": str(path), "truth_count": len(truth), "predicted_count": len(detections),
                     "absolute_count_error": abs(len(truth) - len(detections)), "inference_seconds": seconds,
                     "tp": tp, "fp": fp, "fn": fn})
    result = {"scope": "whole-image generic product detection, measured on supplied test images",
              "device": args.device, "confidence": args.confidence, "matching_iou": args.iou,
              "precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else None,
              "recall": total_tp / (total_tp + total_fn) if total_tp + total_fn else None,
              "visible_facing_count_mae": sum(r["absolute_count_error"] for r in rows) / len(rows),
              "mean_inference_seconds": sum(r["inference_seconds"] for r in rows) / len(rows),
              "per_image": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
