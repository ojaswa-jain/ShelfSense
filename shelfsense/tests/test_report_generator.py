from io import StringIO
import json

import pandas as pd

from src.report_generator import make_report, to_csv, to_json
from src.shelf_analyzer import analyze_shelf


def test_json_and_csv_preserve_evidence():
    report = make_report("image-hash", [], [], [], [], analyze_shelf([], [], []), {"mode": "manual"})
    assert json.loads(to_json(report))["analysis_mode"] == "manual"
    frame = pd.read_csv(StringIO(to_csv(report)))
    assert {"settings", "analysis", "user_corrections", "reference_comparison"} <= set(frame.section)
    assert json.loads(frame[frame.section == "analysis"].iloc[0].value_json)["visible_facings"] == 0


def test_automatic_zero_detection_mode():
    report = make_report("a", [], [], [], [], {}, {"mode": "automatic"})
    assert report["analysis_mode"] == "automatic"
