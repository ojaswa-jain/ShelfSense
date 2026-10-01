from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from src.image_preprocessing import load_rgb, prepare_image


def encoded(mode="RGB", size=(600, 300)):
    out = BytesIO()
    Image.new(mode, size, 128 if mode == "L" else (255, 0, 0)).save(out, format="PNG")
    return out.getvalue()


@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_color_and_grayscale(mode):
    assert load_rgb(encoded(mode)).shape == (300, 600, 3)


def test_invalid_image():
    with pytest.raises(ValueError):
        load_rgb(b"not an image")


def test_reject_disguised_gif():
    out = BytesIO()
    Image.new("RGB", (10, 10)).save(out, format="GIF")
    with pytest.raises(ValueError):
        load_rgb(out.getvalue())


def test_resize_and_mapping():
    result = prepare_image(encoded(), max_side=300)
    assert result.rgb.shape == (150, 300, 3)
    assert result.to_original([10, 20, 100, 120]) == [20, 40, 200, 240]
    assert np.array_equal(result.rgb[0, 0], [255, 0, 0])


def test_exif_orientation():
    out = BytesIO()
    image = Image.new("RGB", (30, 20))
    exif = image.getexif()
    exif[274] = 6
    image.save(out, format="JPEG", exif=exif)
    assert load_rgb(out.getvalue()).shape == (30, 20, 3)


def test_transparency():
    out = BytesIO()
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(out, format="PNG")
    assert (load_rgb(out.getvalue()) == 255).all()
