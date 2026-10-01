from io import BytesIO

from PIL import Image
import pytest

from src.catalogue_manager import CatalogueManager


def reference():
    out = BytesIO()
    Image.new("RGB", (10, 20), "red").save(out, format="PNG")
    return out.getvalue()


def test_add_update_duplicate_and_fingerprint(tmp_path):
    manager = CatalogueManager(tmp_path)
    manager.save_product("A", "Tea", references=[reference()])
    first = manager.fingerprint()
    with pytest.raises(ValueError):
        manager.save_product("A", "Tea", references=[reference()])
    manager.save_product("A", "Tea updated", minimum_facings=2, update=True)
    assert manager.load()[0]["minimum_facings"] == 2
    assert manager.fingerprint() != first


@pytest.mark.parametrize("pid,name,target", [("../bad", "Tea", 1), ("A", "", 1), ("A", "Tea", -1)])
def test_invalid_fields(tmp_path, pid, name, target):
    with pytest.raises(ValueError):
        CatalogueManager(tmp_path).save_product(pid, name, minimum_facings=target, references=[reference()])


def test_missing_reference_and_corrupted_catalogue(tmp_path):
    manager = CatalogueManager(tmp_path)
    with pytest.raises(ValueError):
        manager.save_product("A", "Tea")
    manager.path.write_text("invalid")
    with pytest.raises(ValueError):
        manager.load()
