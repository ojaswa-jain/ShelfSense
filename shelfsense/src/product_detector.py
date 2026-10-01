"""Optional local Ultralytics detector. No implicit model download or fake boxes."""
from pathlib import Path
from uuid import uuid4

import numpy as np


class ProductDetector:
    def __init__(self, weights: str, device: str = "cpu"):
        path = Path(weights).expanduser().resolve()
        if not path.is_file() or path.suffix != ".pt":
            raise ValueError("Provide an existing, trusted Ultralytics detection .pt file.")
        from ultralytics import YOLO, settings
        settings.update({"sync": False})
        self.model = YOLO(str(path), task="detect")
        if self.model.task != "detect":
            raise ValueError("Only axis-aligned object detection weights are supported.")
        self.device, self.name = device, str(path)

    def detect(self, rgb: np.ndarray, confidence: float = 0.35,
               iou: float = 0.5) -> list[dict]:
        if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
            raise ValueError("Confidence and IoU must lie between 0 and 1.")
        # Ultralytics accepts BGR arrays. Its output is mapped back to this input size.
        prediction = self.model.predict(source=rgb[:, :, ::-1].copy(), conf=confidence,
                                        iou=iou, device=self.device, imgsz=960,
                                        agnostic_nms=True, max_det=500, verbose=False)[0]
        height, width = rgb.shape[:2]
        result = []
        for box in prediction.boxes:
            coords = box.xyxy[0].cpu().tolist()
            coords = [max(0.0, min(coords[0], width)), max(0.0, min(coords[1], height)),
                      max(0.0, min(coords[2], width)), max(0.0, min(coords[3], height))]
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                continue
            result.append({"id": uuid4().hex[:12], "box": coords,
                           "confidence": float(box.conf[0]), "source": "model",
                           "detector_class": str(self.model.names[int(box.cls[0])]),
                           "product_id": None, "status": "Unknown product",
                           "similarity": None, "candidates": []})
        return result
