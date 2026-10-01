"""A small local JSON catalogue with validated, content-addressed references."""
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

from src.image_preprocessing import load_rgb


class CatalogueManager:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.images = self.root / "reference_images"
        self.images.mkdir(exist_ok=True)
        self.path = self.root / "catalogue.json"

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            entries = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(entries, list):
                raise ValueError("Catalogue must contain a JSON list.")
            ids = set()
            for entry in entries:
                self._validate(entry)
                if entry["product_id"] in ids:
                    raise ValueError("Duplicate product ID in catalogue.")
                ids.add(entry["product_id"])
            return entries
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError("Cannot read catalogue; restore valid catalogue.json.") from error

    def _validate(self, entry: dict) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", entry["product_id"]):
            raise ValueError("Product ID: 1–64 letters, digits, underscores or hyphens.")
        if not isinstance(entry.get("name"), str) or not entry["name"].strip():
            raise ValueError("A product name is required.")
        target = entry.get("minimum_facings")
        if target is not None and (type(target) is not int or target < 0):
            raise ValueError("Minimum facings must be a non-negative integer.")
        for name in entry.get("reference_images", []):
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".png"):
                raise ValueError("Invalid reference image filename.")

    def save_product(self, product_id: str, name: str, brand: str = "",
                     expected_region: str = "", minimum_facings: int | None = None,
                     references: list[bytes] | None = None, update: bool = False) -> dict:
        entries = self.load()
        old = next((p for p in entries if p["product_id"] == product_id), None)
        if old and not update:
            raise ValueError("Product ID already exists. Select update to edit it.")
        if update and old is None:
            raise ValueError("Cannot update a product that does not exist.")
        entry = {"product_id": product_id, "name": name.strip(), "brand": brand.strip(),
                 "expected_region": expected_region.strip() or None,
                 "minimum_facings": minimum_facings,
                 "reference_images": list(old["reference_images"]) if old else []}
        self._validate(entry)
        # Decode every new reference before writing anything.
        decoded = [(hashlib.sha256(raw).hexdigest() + ".png", load_rgb(raw))
                   for raw in (references or [])]
        if not entry["reference_images"] and not decoded:
            raise ValueError("Register at least one reference image.")
        for filename, rgb in decoded:
            Image.fromarray(rgb).save(self.images / filename)
            if filename not in entry["reference_images"]:
                entry["reference_images"].append(filename)
        entries = [p for p in entries if p["product_id"] != product_id] + [entry]
        self._write(entries)
        return entry

    def delete(self, product_id: str) -> None:
        self._write([p for p in self.load() if p["product_id"] != product_id])
        # Keep reference files: references may be shared; no destructive cleanup.

    def _write(self, entries: list[dict]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)

    def fingerprint(self) -> str:
        digest = hashlib.sha256(json.dumps(self.load(), sort_keys=True).encode())
        for entry in self.load():
            for name in entry["reference_images"]:
                path = self.images / name
                if not path.is_file():
                    raise ValueError(f"Missing catalogue image: {name}")
                digest.update(path.read_bytes())
        return digest.hexdigest()
