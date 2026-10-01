import json

import pytest

from src.settings import load_settings


def test_invalid_config(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"max_side": 999}))
    with pytest.raises(ValueError):
        load_settings(path)
