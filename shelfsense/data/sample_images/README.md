# Synthetic geometry example — not model output

`synthetic_shelf.png` is a simple rendered fixture, deliberately labelled SYNTHETIC. Its three rectangles are manually specified visible-facing annotations, not detector predictions.

Upload it in manual mode, apply region `Shelf 1 = [20,40,780,280]`, enter the three boxes in `synthetic_boxes.json`, apply corrections, and analyse. Keep matching disabled. Expected: 3 visible facings, 3 unknown identities, 48.2456% estimated visual occupancy, 2 potential gaps at the default 8% width threshold.

`synthetic_annotated.png` and `synthetic_audit.json` are computed from those manual annotations using the actual audit engine. They demonstrate geometry and report structure only. They are not retail model evaluation results.

For real evaluation, supply your own consented/licensed photographs and ground truth. Use separate capture sessions/shelf arrangements for training, threshold calibration and testing.
