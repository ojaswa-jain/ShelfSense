# Evaluation protocol and validation status

## What has and has not been measured

The supplied test suite checks deterministic algorithms and Streamlit interactions with synthetic inputs. Optional neural model inference, training, Windows installation and real-shop accuracy have not been run in this build environment. Do not cite synthetic unit tests as computer-vision accuracy.

No real-image precision, recall, MAE, matching accuracy, unknown rejection, gap accuracy or CPU inference benchmark is supplied. The detector evaluation script computes actual measurements only after you provide images, complete labels and weights.

## Data partitioning

- Create training, validation/calibration and test partitions grouped by capture session or shelf arrangement. Record session IDs and image hashes in a manifest.
- Keep near-identical frames together. Never randomly split a video into train/test frames.
- Build product references from separate registration captures, not test crops. Include unseen products and visually similar variants in the held-out test set.
- Fix detection confidence, NMS IoU, similarity threshold, ambiguity margin and gap threshold using validation data. Freeze them before testing.
- Define annotation policy for occluded fronts and partly visible packages. Have another person review ground truth when possible.

## Metrics

| Metric | Procedure |
|---|---|
| Detection precision/recall | Confidence-ranked one-to-one greedy matching at IoU >= 0.5; precision TP/(TP+FP), recall TP/(TP+FN). Report confidence and IoU thresholds. The script reports null for undefined denominators |
| Visible-facing count MAE | Mean absolute difference between predicted and ground-truth facing counts. Script measures whole-image counts; for shelf-region results apply identical region rules to both annotations |
| Known-product matching accuracy | On held-out ground-truth crops, fraction of known-product crops assigned the correct ID. Rejected/ambiguous known samples count as incorrect. Separately report accuracy among accepted matches and acceptance coverage |
| Unknown rejection | Among crops with products absent from the catalogue, fraction receiving no ID (Unknown or Needs verification). Also report false acceptance rate and known-product rejection rate |
| Potential-gap precision/recall | Manually mark visibly empty horizontal intervals within fixed shelf rows. Match predicted and true intervals one-to-one by interval IoU >= 0.5. Report TP/(TP+FP), TP/(TP+FN), and exclude physically unobservable/ambiguous intervals using a predefined policy |
| CPU inference time | Record processor, RAM, package/model versions, image size, crop count and threads. Separate cold model load, reference embedding time, detector time, crop matching time and total audit time. Warm once before steady-state measurements; use median and p95 over multiple images/runs |

The detector script handles the first two metrics and detector wall time. Matching and gap metrics require a labelled evaluation set and a separate evaluation notebook/procedure; this release documents their protocol rather than fabricating results. For matching, export audit JSON and compare `corrected_result` only when evaluating a human-assisted workflow; use uncorrected initial results for an automatic benchmark. Report human-assisted and automatic performance separately.

## Record each experiment

Keep dataset split manifest, model checkpoint hash, catalogue snapshot, settings, per-image predictions, ground truth and metric denominators. Store outputs under `outputs/`; do not commit private images or large weights to Git. Report both failure examples and aggregate results. If a test set is small, include uncertainty intervals and avoid generalizing to other shops.

## Functional validation

Run `python -m pytest -q` from the project root. The tests need only the base requirements and no downloaded model weights. They include a Streamlit AppTest that adds a manual box, applies it, analyses the shelf and verifies state invalidation when the source changes.

## Build verification result

35 tests passed on Python 3.12.14 (Linux), using NumPy 2.2.6, Pillow 11.2.1, pandas 2.2.3, OpenCV 4.11.0 (headless wheel in the build environment) and Streamlit 1.45.1. Python source compilation and both script help entry points also passed. Windows instructions target Python 3.11 and the GUI-capable OpenCV wheel; those platform-specific installation steps were not executed here.
