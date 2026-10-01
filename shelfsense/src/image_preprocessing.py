"""Validated RGB loading, mild preprocessing and explicit coordinate mapping."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass
class PreparedImage:
    original: np.ndarray
    rgb: np.ndarray
    detection_rgb: np.ndarray
    blur_score: float

    def to_original(self, box: list[float]) -> list[float]:
        """Map working coordinates to the EXIF-oriented original image."""
        h, w = self.rgb.shape[:2]
        oh, ow = self.original.shape[:2]
        return [box[0] * ow / w, box[1] * oh / h,
                box[2] * ow / w, box[3] * oh / h]


def load_rgb(source: bytes | str | Path) -> np.ndarray:
    """Read actual JPEG/PNG content, orient, composite alpha onto white."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            stream = BytesIO(source) if isinstance(source, bytes) else source
            with Image.open(stream) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG image content is supported.")
                if image.width * image.height > 24_000_000:
                    raise ValueError("Image exceeds the 24 megapixel limit.")
                image.load()
                oriented = ImageOps.exif_transpose(image)
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    base = Image.new("RGBA", rgba.size, "white")
                    oriented = Image.alpha_composite(base, rgba)
                return np.array(oriented.convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("Invalid, corrupted, or excessively large image.") from error


def prepare_image(source: bytes | str | Path, max_side: int = 1600,
                  contrast: bool = False) -> PreparedImage:
    if not 256 <= max_side <= 4096:
        raise ValueError("max_side must be between 256 and 4096.")
    original = load_rgb(source)
    h, w = original.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    rgb = cv2.resize(original, (max(1, round(w * scale)), max(1, round(h * scale))),
                     interpolation=cv2.INTER_AREA) if scale < 1 else original.copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    enhanced = rgb.copy()
    if contrast:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return PreparedImage(original, rgb, enhanced, blur)
