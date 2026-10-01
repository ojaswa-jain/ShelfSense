from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_opens_without_models():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "ShelfSense"
    assert len(app.tabs) == 5


def test_manual_audit_and_source_invalidation():
    code = '''
import numpy as np
import streamlit as st
from app import audit_panel
from src.image_preprocessing import PreparedImage
from src.catalogue_manager import CatalogueManager
from pathlib import Path
settings = dict(mode='manual', weights='', matching=False, device='cpu', allow_download=False,
                detector_confidence=.35, nms_iou=.5, similarity_threshold=.85,
                ambiguity_margin=.05, minimum_gap_fraction=.08, blur_warning_threshold=80)
rgb = np.zeros((300,600,3), dtype=np.uint8)
image_id = st.text_input('Source ID', 'image-one')
report = audit_panel(PreparedImage(rgb,rgb,rgb,100), image_id, 'test', settings, [], CatalogueManager(Path('data')))
'''
    app = AppTest.from_string(code).run(timeout=20)
    assert not app.exception
    # Simulate a table edit using Streamlit's own data_editor state schema.
    editor_key = "ui_test_boxes_image-one_600x300_0"
    app.session_state[editor_key] = {
        "edited_rows": {}, "deleted_rows": [],
        "added_rows": [{"x1": 10.0, "y1": 10.0, "x2": 200.0, "y2": 290.0}]}
    app.button[1].click().run()
    assert not app.exception
    app.button[2].click().run()
    assert not app.exception
    assert app.metric[0].value == "1"
    app.text_input[0].set_value("image-two").run()
    assert not app.exception
    assert app.session_state["audit_test"]["detections"] == []
    assert not app.session_state["audit_test"]["ready"]
