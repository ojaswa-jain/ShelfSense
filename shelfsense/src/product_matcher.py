"""Optional ResNet18 embeddings and conservative, per-product cosine matching."""
from pathlib import Path

import numpy as np
from PIL import Image


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or not np.all(np.isfinite(vector)) or norm <= 1e-12:
        raise ValueError("Embedding must be a finite non-zero vector.")
    return vector / norm


def match_embedding(query: np.ndarray, references: dict[str, list[np.ndarray]],
                    threshold: float = 0.85, margin: float = 0.05) -> dict:
    if not -1 <= threshold <= 1 or not 0 <= margin <= 2:
        raise ValueError("Invalid similarity threshold or ambiguity margin.")
    query = normalize(query)
    candidates = []
    for pid, vectors in references.items():
        if vectors:
            score = max(float(np.clip(query @ normalize(v), -1, 1)) for v in vectors)
            candidates.append({"product_id": pid, "similarity": score})
    candidates.sort(key=lambda row: row["similarity"], reverse=True)
    base = {"product_id": None, "status": "Unknown product", "similarity": None,
            "candidates": candidates[:3]}
    if not candidates:
        return base
    score = candidates[0]["similarity"]
    base["similarity"] = score
    if score < threshold:
        return base
    if len(candidates) > 1 and score - candidates[1]["similarity"] < margin:
        base["status"] = "Needs verification"
        return base
    return {**base, "product_id": candidates[0]["product_id"], "status": "Reference match"}


class EmbeddingModel:
    """512-D pooled ImageNet features; not a retail identity classifier."""
    name = "torchvision ResNet18 / IMAGENET1K_V1 / 512-D pooled features"

    def __init__(self, device: str = "cpu", allow_download: bool = False):
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        if device != "cpu" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; choose CPU.")
        self.torch, self.device = torch, device
        weights = ResNet18_Weights.IMAGENET1K_V1
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
        if not checkpoint.exists() and not allow_download:
            raise ValueError("ResNet18 weights are not cached. Enable the initial download in Settings or disable matching.")
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(device)
        self.transform = weights.transforms()

    def encode(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.size == 0:
            raise ValueError("Cannot embed an empty crop.")
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            vector = self.model(tensor)[0].cpu().numpy()
        return normalize(vector)


def build_reference_embeddings(catalogue: list[dict], images_dir: Path,
                               model: EmbeddingModel) -> dict[str, list[np.ndarray]]:
    from src.image_preprocessing import load_rgb
    return {p["product_id"]: [model.encode(load_rgb(images_dir / name))
                              for name in p["reference_images"]] for p in catalogue}


def identify_products(rgb: np.ndarray, detections: list[dict], model: EmbeddingModel,
                      references: dict[str, list[np.ndarray]], threshold: float,
                      margin: float) -> list[dict]:
    from copy import deepcopy
    result = deepcopy(detections)
    for detection in result:
        x1, y1 = np.floor(detection["box"][:2]).astype(int)
        x2, y2 = np.ceil(detection["box"][2:]).astype(int)
        detection.update(match_embedding(model.encode(rgb[y1:y2, x1:x2]), references,
                                         threshold, margin))
    return result
