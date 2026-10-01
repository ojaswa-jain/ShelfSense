"""RGB overlays with distinct regions, facings and uncertain gap markings."""
import cv2
import numpy as np


def annotate(rgb: np.ndarray, detections: list[dict], regions: list[dict],
             analysis: dict | None = None) -> np.ndarray:
    image = rgb.copy()
    if analysis:
        overlay = image.copy()
        for row in analysis["regions"]:
            for gap in row["potential_gaps"]:
                x1, y1, x2, y2 = map(round, gap["box"])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 180, 55), -1)
        image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
    for region in regions:
        x1, y1, x2, y2 = map(round, region["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (70, 170, 245), 2)
        cv2.putText(image, region["name"], (x1 + 4, min(y2 - 4, y1 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 170, 245), 2)
    for index, detection in enumerate(detections, 1):
        x1, y1, x2, y2 = map(round, detection["box"])
        color = (45, 210, 160) if detection.get("product_id") else (255, 180, 55)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f'{index}: {detection.get("product_id") or "Unknown"}'
        if detection.get("confidence") is not None:
            label += f' det={detection["confidence"]:.2f}'
        cv2.putText(image, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    return image
