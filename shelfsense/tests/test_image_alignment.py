import cv2
import numpy as np

from src.image_alignment import align_reference, regions_covered, to_reference_original


def test_blank_rejected():
    image = np.zeros((400, 600, 3), np.uint8)
    assert not align_reference(image, image)["success"]


def test_textured_small_translation():
    image = np.random.default_rng(42).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 6], [0, 1, 4]]), (640, 480))
    result = align_reference(image, shifted)
    assert result["success"]
    assert regions_covered([{"box": [20, 20, 620, 460]}], result["valid_mask"])
    assert not regions_covered([{"box": [0, 0, 640, 480]}], result["valid_mask"])


def test_inverse_coordinate_mapping():
    mapped = to_reference_original([{"id": "a", "box": [10, 20, 30, 40]}],
                                    [[1, 0, 5], [0, 1, 10], [0, 0, 1]], (100, 100), (200, 200))
    assert mapped[0]["polygon"] == [[10, 20], [50, 20], [50, 60], [10, 60]]
