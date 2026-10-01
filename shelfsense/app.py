"""Run with: python -m streamlit run app.py"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time

import pandas as pd
from PIL import Image
import streamlit as st

from src.annotation_editor import apply_corrections, editor_rows, validate_regions
from src.catalogue_manager import CatalogueManager
from src.image_alignment import align_reference, regions_covered, to_reference_original
from src.image_preprocessing import prepare_image
from src.product_detector import ProductDetector
from src.product_matcher import EmbeddingModel, build_reference_embeddings, identify_products
from src.report_generator import DISCLAIMER, make_report, to_csv, to_json
from src.shelf_analyzer import analyze_shelf, compare_audits
from src.settings import load_settings
from src.visualization import annotate


ROOT = Path(__file__).resolve().parent
BOX_COLUMNS = ["id", "x1", "y1", "x2", "y2", "product_id", "remove"]
REGION_COLUMNS = ["name", "x1", "y1", "x2", "y2"]


@st.cache_data(show_spinner=False)
def prepared(raw: bytes, max_side: int, contrast: bool):
    return prepare_image(raw, max_side, contrast)


@st.cache_resource(show_spinner=False)
def detector_resource(path: str, stamp: int, device: str):
    return ProductDetector(path, device)


@st.cache_resource(show_spinner=False)
def embedding_resource(device: str, allow_download: bool):
    return EmbeddingModel(device, allow_download)


@st.cache_data(show_spinner=False)
def reference_vectors(fingerprint: str, catalogue_json: str, device: str, _model):
    return build_reference_embeddings(json.loads(catalogue_json), ROOT / "data/reference_images", _model)


def records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def png_bytes(rgb) -> bytes:
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def clear_audits() -> None:
    for key in list(st.session_state):
        if key.startswith(("audit_", "ui_", "alignment_", "confirm_")):
            del st.session_state[key]


def settings_panel(defaults: dict) -> dict:
    with st.sidebar:
        st.markdown("### Audit controls")
        mode = st.radio("Analysis mode", ["manual", "automatic"],
                        help="Automatic mode requires your own compatible retail detector weights.")
        path = st.text_input("Local detector .pt path", "models/retail_best.pt", disabled=mode == "manual")
        confidence = st.slider("Detector confidence", 0.05, 0.95, defaults["detector_confidence"], 0.05)
        matching = st.checkbox("Enable reference-product matching", value=False)
        similarity = st.slider("Cosine similarity threshold", 0.0, 1.0, defaults["similarity_threshold"], 0.01)
        margin = st.slider("Best/second-best product margin", 0.0, 0.5, defaults["ambiguity_margin"], 0.01)
        gap = st.slider("Minimum gap (% of region width)", 1, 50, round(defaults["minimum_gap_fraction"] * 100))
        st.caption("Similarity is not a probability. Thresholds need local validation.")
        if st.button("Clear / reset session", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    return {"mode": mode, "weights": path, "detector_confidence": confidence,
            "similarity_threshold": similarity, "ambiguity_margin": margin,
            "minimum_gap_fraction": gap / 100, "matching": matching}


def catalogue_panel(manager: CatalogueManager, catalogue: list[dict]) -> None:
    st.subheader("Your product reference catalogue")
    st.caption("Start with 5–10 products. Use tightly cropped front views in several lighting conditions.")
    if catalogue:
        st.dataframe(pd.DataFrame(catalogue).drop(columns="reference_images"), hide_index=True,
                     use_container_width=True)
    choices = ["Add a new product"] + [p["product_id"] for p in catalogue]
    selected = st.selectbox("Add or update", choices)
    existing = next((p for p in catalogue if p["product_id"] == selected), {})
    with st.form(f"catalogue_{selected}", clear_on_submit=False):
        col1, col2 = st.columns(2)
        pid = col1.text_input("Product ID", existing.get("product_id", ""), disabled=bool(existing))
        name = col2.text_input("Product name", existing.get("name", ""))
        brand = col1.text_input("Brand (optional)", existing.get("brand", ""))
        region = col2.text_input("Expected shelf region (optional)", existing.get("expected_region") or "")
        use_target = col1.checkbox("Set minimum visible-facing target", existing.get("minimum_facings") is not None)
        target = col2.number_input("Target", min_value=0, max_value=1000,
                                   value=existing.get("minimum_facings") or 0, step=1)
        refs = st.file_uploader("Add reference images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if st.form_submit_button("Save product", type="primary"):
            try:
                manager.save_product(pid.strip(), name, brand, region, int(target) if use_target else None,
                                     [f.getvalue() for f in refs], update=bool(existing))
                clear_audits()
                st.rerun()
            except ValueError as error:
                st.error(str(error))
    if existing:
        st.caption(f'{len(existing["reference_images"])} reference images stored. Updates append new views.')
        cols = st.columns(min(4, len(existing["reference_images"])))
        for index, filename in enumerate(existing["reference_images"]):
            path = manager.images / filename
            if path.exists():
                cols[index % len(cols)].image(str(path), width=140)
        if st.checkbox("Confirm removal from catalogue", key=f"delete_{selected}"):
            if st.button("Remove selected product"):
                manager.delete(selected)
                clear_audits()
                st.rerun()


def audit_panel(image, image_id: str, prefix: str, settings: dict, catalogue: list[dict],
                manager: CatalogueManager, fixed_regions: list[dict] | None = None) -> dict | None:
    """The same fully editable audit workflow serves current and reference images."""
    h, w = image.rgb.shape[:2]
    widget_id = f"{image_id}_{w}x{h}"
    state_key = f"audit_{prefix}"
    state = st.session_state.get(state_key)
    if state is None or state["image_id"] != image_id or state.get("working_size") != [w, h]:
        state = {"image_id": image_id, "detections": [], "original": [], "corrections": [],
                 "regions": [{"name": "Shelf 1", "box": [0, 0, w, h]}],
                 "ready": False, "revision": 0, "working_size": [w, h]}
        st.session_state[state_key] = state
    if fixed_regions is not None and state["regions"] != fixed_regions:
        state["regions"] = deepcopy(fixed_regions)
        state["ready"] = False
        state["revision"] += 1
    signature = json.dumps(settings, sort_keys=True)
    if state.get("settings_signature") != signature and state["ready"]:
        st.warning("Controls changed. Analyse again to use these settings; exports are paused.")
    st.caption(f"Working image: {w} × {h} pixels. All editor coordinates use this size; origin is top-left.")
    if image.blur_score < settings["blur_warning_threshold"]:
        st.warning(f"Image may be blurred (Laplacian variance {image.blur_score:.1f}). Inspect small text and boxes.")
    current_analysis = analyze_shelf(state["detections"], state["regions"], catalogue,
                                     settings["minimum_gap_fraction"])
    left, right = st.columns(2)
    left.image(image.rgb, caption="Original colour image (resized for analysis)", use_container_width=True)
    right.image(annotate(image.rgb, state["detections"], state["regions"], current_analysis),
                caption="Blue: shelf regions · Green: identified · Amber: uncertain / potential gap",
                use_container_width=True)
    with st.expander("1 · Define shelf rows / regions", expanded=not state["ready"]):
        st.caption("Use one narrow rectangle per shelf row. Full-image occupancy can be misleading.")
        region_rows = [{"name": r["name"], **dict(zip(REGION_COLUMNS[1:], r["box"]))} for r in state["regions"]]
        if fixed_regions is not None:
            st.dataframe(region_rows, hide_index=True)
            st.caption("Regions inherited from the current shelf after alignment.")
        else:
            edited = st.data_editor(pd.DataFrame(region_rows), num_rows="dynamic", hide_index=True,
                                    key=f"ui_{prefix}_regions_{widget_id}_{state['revision']}", use_container_width=True)
            if st.button("Apply shelf regions", key=f"regions_{prefix}"):
                try:
                    state["regions"] = validate_regions(records(edited), w, h)
                    state["ready"] = False
                    state["revision"] += 1
                    st.rerun()
                except (ValueError, TypeError) as error:
                    st.error(str(error))
    with st.expander("2 · Add, adjust or remove product boxes", expanded=True):
        st.caption("Add a table row for each visible facing. Enter x1, y1, x2, y2. Leave ID blank for new boxes. "
                   "Remove via the checkbox or row deletion. Apply edits before analysing.")
        frame = pd.DataFrame(editor_rows(state["detections"]), columns=BOX_COLUMNS)
        box_rows = st.data_editor(frame, num_rows="dynamic", hide_index=True,
                                  disabled=["id"], use_container_width=True,
                                  key=f"ui_{prefix}_boxes_{widget_id}_{state['revision']}",
                                  column_config={
                                      "product_id": st.column_config.SelectboxColumn("Product ID", options=[""] + [p["product_id"] for p in catalogue]),
                                      "remove": st.column_config.CheckboxColumn("Remove", default=False),
                                      **{key: st.column_config.NumberColumn(key, min_value=0.0, step=1.0)
                                         for key in ["x1", "y1", "x2", "y2"]}})
        if st.button("Apply box / identity corrections", key=f"apply_{prefix}"):
            try:
                result, events = apply_corrections(state["detections"], records(box_rows), w, h,
                                                   {p["product_id"] for p in catalogue})
                state["detections"] = result
                state["corrections"].extend(events)
                state["revision"] += 1
                state["timestamp"] = datetime.now(timezone.utc).isoformat()
                st.rerun()
            except (ValueError, TypeError) as error:
                st.error(f"Edits not applied: {error}")
    if st.button("Analyse Shelf", type="primary", key=f"analyse_{prefix}"):
        try:
            progress = st.progress(0, text="Preparing audit…")
            started = time.perf_counter()
            # Commit only after all inference steps succeed; preserve last result on errors.
            result = deepcopy(state["detections"])
            model_name = "Manual annotations (no detector)"
            model_digest = None
            if settings["mode"] == "automatic":
                path = Path(settings["weights"]).expanduser()
                path = path if path.is_absolute() else ROOT / path
                if not path.is_file():
                    raise ValueError("Detector file not found. Supply retail .pt weights or choose manual mode.")
                detector = detector_resource(str(path.resolve()), path.stat().st_mtime_ns, settings["device"])
                result = detector.detect(image.detection_rgb, settings["detector_confidence"], settings["nms_iou"])
                model_name = detector.name
                model_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            progress.progress(40, text="Checking product references…")
            if settings["matching"]:
                if not catalogue:
                    raise ValueError("Register products first, or disable reference-product matching.")
                model = embedding_resource(settings["device"], settings["allow_download"])
                refs = reference_vectors(manager.fingerprint(), json.dumps(catalogue), settings["device"], model)
                # Respect human identity assignments in manual mode.
                pending = [d for d in result if d["status"] != "Manual identity"]
                identified = identify_products(image.rgb, pending, model, refs,
                                                settings["similarity_threshold"], settings["ambiguity_margin"])
                by_id = {d["id"]: d for d in identified}
                result = [by_id.get(d["id"], d) for d in result]
            elif settings["mode"] == "manual":
                for detection in result:
                    if detection["status"] != "Manual identity":
                        detection.update(product_id=None, status="Unknown product", similarity=None, candidates=[])
            progress.progress(85, text="Calculating visible facings and visual occupancy…")
            if settings["mode"] == "automatic" or not state["ready"] and not state["original"]:
                state["original"] = deepcopy(result)
                state["initial_detection_mode"] = settings["mode"]
                state["initial_detector_model"] = model_name
                state["initial_weights_sha256"] = model_digest
            state.update(detections=result, ready=True,
                         settings_signature=signature,
                         run_settings={**settings, "detector_model": model_name,
                                       "weights_sha256": model_digest,
                                       "initial_detection_mode": state.get("initial_detection_mode", "manual"),
                                       "initial_detector_model": state.get("initial_detector_model"),
                                       "initial_weights_sha256": state.get("initial_weights_sha256"),
                                       "embedding_model": EmbeddingModel.name if settings["matching"] else None,
                                       "elapsed_seconds": time.perf_counter() - started},
                         timestamp=datetime.now(timezone.utc).isoformat())
            if settings["mode"] == "automatic":
                state["corrections"] = []
            state["revision"] += 1
            progress.progress(100, text="Audit ready")
            st.rerun()
        except (ValueError, OSError, ImportError, RuntimeError) as error:
            st.error(f"Analysis could not finish: {error}. Manual mode remains available.")
    if not state["ready"] or state.get("settings_signature") != signature:
        st.info("Apply regions and boxes, then select Analyse Shelf to produce a report.")
        return None
    analysis = analyze_shelf(state["detections"], state["regions"], catalogue, settings["minimum_gap_fraction"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Visible facings", analysis["visible_facings"])
    col2.metric("Unidentified facings", analysis["unknown_product_count"])
    col3.metric("Potential gaps", sum(len(row["potential_gaps"]) for row in analysis["regions"]))
    for row in analysis["regions"]:
        st.write(f'**{row["region"]}** · {row["visible_facings"]} visible facings · '
                 f'{row["estimated_visual_occupancy"]:.1%} estimated visual occupancy')
        st.progress(min(1.0, max(0.0, row["estimated_visual_occupancy"])))
    if analysis["outside_region_count"]:
        st.warning(f'{analysis["outside_region_count"]} boxes have centres outside audited regions and are excluded from facing counts.')
    if state["detections"]:
        st.dataframe(pd.DataFrame([{k: d.get(k) for k in ["id", "product_id", "status", "confidence", "similarity", "source"]}
                                   for d in state["detections"]]), hide_index=True, use_container_width=True)
        with st.expander("Matching evidence / candidate rankings"):
            st.json([{k: d.get(k) for k in ["id", "candidates"]} for d in state["detections"]])
    if analysis["targets"]:
        st.caption("Display-facing targets: shortfalls are inspection prompts, not replenishment quantities.")
        st.dataframe(analysis["targets"], hide_index=True, use_container_width=True)
    if analysis["possible_misplacements"]:
        st.warning("Possible misplaced products — verify the expected region and identity.")
        st.dataframe(analysis["possible_misplacements"], hide_index=True)
    st.caption(DISCLAIMER)
    report = make_report(image_id, state["detections"], state["original"], state["corrections"],
                         state["regions"], analysis, state["run_settings"], catalogue=catalogue,
                         coordinate_info={"system": "EXIF-oriented resized RGB, xyxy, half-open",
                                          "working_size": [w, h],
                                          "oriented_original_size": [image.original.shape[1], image.original.shape[0]],
                                          "original_boxes": [{"id": d["id"], "box": image.to_original(d["box"])}
                                                             for d in state["detections"]]},
                         timestamp=state["timestamp"])
    st.download_button("Download annotated PNG", png_bytes(annotate(image.rgb, state["detections"], state["regions"], analysis)),
                       f"{prefix}_annotated.png", "image/png", key=f"png_{prefix}")
    return report


def main() -> None:
    st.set_page_config(page_title="ShelfSense · Kirana Shelf Auditor", page_icon="🛒", layout="wide")
    st.title("ShelfSense")
    st.markdown("**Kirana Shelf Auditor** · See the shelf. Check the evidence. Inspect the gaps.")
    st.caption("Local image processing · Visible facings, not total inventory · Manual mode needs no model weights")
    try:
        defaults = load_settings(ROOT / "config/default_settings.json")
    except ValueError as error:
        st.error(str(error))
        st.stop()
    settings = settings_panel(defaults)
    tabs = st.tabs(["Shelf Audit", "Product Catalogue", "Reference Comparison", "Reports", "Settings"])
    with tabs[4]:
        st.subheader("Processing and model settings")
        settings["device"] = st.selectbox("Inference device", ["cpu", "cuda:0"])
        settings["allow_download"] = st.checkbox("Allow initial ResNet18 weight download (~45 MB)", False)
        settings["max_side"] = st.select_slider("Maximum analysis image side", [800, 1200, 1600, 2048], value=defaults["max_side"])
        settings["contrast"] = st.checkbox("Mild CLAHE contrast for detection only", False)
        settings["nms_iou"] = st.slider("Detection NMS IoU", 0.1, 0.9, defaults["nms_iou"], 0.05)
        settings["blur_warning_threshold"] = st.number_input("Blur warning threshold (heuristic)", 0.0, 1000.0, defaults["blur_warning_threshold"])
        st.info("ResNet18 is a generic visual descriptor, not a product identity expert. Similar packaging can match incorrectly. "
                "YOLO requires retail-trained detection weights; COCO weights do not cover all packages. "
                "Load only trusted .pt files. Matching and detection may be slow on CPU.")
        st.caption("Reference embeddings and models are cached in memory. Catalogue edits invalidate audits. "
                   "No uploaded images are sent to model services. See README for licensing and offline setup.")
    manager = CatalogueManager(ROOT / "data")
    try:
        catalogue = manager.load()
    except ValueError as error:
        st.error(str(error))
        st.stop()
    with tabs[1]:
        catalogue_panel(manager, catalogue)
    report, image, raw = None, None, None
    with tabs[0]:
        upload = st.file_uploader("Upload current shelf image", type=["jpg", "jpeg", "png"], key="current_upload")
        if upload:
            raw = upload.getvalue()
            try:
                image = prepared(raw, settings["max_side"], settings["contrast"])
                image_id = hashlib.sha256(raw).hexdigest()
                report = audit_panel(image, image_id, "current", settings, catalogue, manager)
            except ValueError as error:
                st.error(str(error))
        else:
            st.session_state.pop("audit_current", None)
            st.info("Upload one shelf photo to begin. Manual mode: define a row, add product boxes, then analyse.")
    comparison = {"status": "Not performed"}
    with tabs[2]:
        ref_upload = st.file_uploader("Upload desired / reference shelf image", type=["jpg", "jpeg", "png"], key="reference_upload")
        if ref_upload and image is not None:
            try:
                ref_raw = ref_upload.getvalue()
                ref = prepared(ref_raw, settings["max_side"], settings["contrast"])
                pair = hashlib.sha256(raw + ref_raw + str(settings["max_side"]).encode()).hexdigest()
                key = f"alignment_{pair}"
                choice = st.radio("Comparison method", ["Automatic alignment", "Manual corresponding regions", "Independent reference audit"])
                selected_image, fixed, ready_to_compare = ref, None, False
                if choice == "Automatic alignment":
                    if st.button("Attempt alignment"):
                        with st.spinner("Matching shelf features…"):
                            st.session_state[key] = align_reference(ref.rgb, image.rgb)
                    alignment = st.session_state.get(key)
                    if alignment:
                        st.json({k: alignment[k] for k in ["success", "reason", "metrics"]})
                    if alignment and alignment["success"]:
                        st.image(alignment["warped"], caption="Aligned reference — check shelf correspondence", use_container_width=True)
                        if report and regions_covered(report["regions"], alignment["valid_mask"]):
                            ready_to_compare = st.checkbox("I verified the aligned shelf regions correspond", key=f"confirm_{pair}")
                            from src.image_preprocessing import PreparedImage
                            selected_image = PreparedImage(alignment["warped"], alignment["warped"], alignment["warped"], ref.blur_score)
                            fixed = report["regions"]
                        else:
                            st.warning("Complete the current audit and ensure its regions are fully inside valid reference coverage.")
                    else:
                        st.info("No accepted alignment. You can still audit the reference independently below, or select manual corresponding regions.")
                elif choice == "Manual corresponding regions":
                    st.info("Define reference rectangles below with exactly the same names as current regions. Compare only the same physical shelf area.")
                    ready_to_compare = st.checkbox("I verified these regions show the same physical shelf areas", key=f"confirm_manual_{pair}")
                identity = hashlib.sha256(ref_raw).hexdigest() + (f"_aligned_{pair}" if fixed else "_native")
                reference_report = audit_panel(selected_image, identity, "reference", settings, catalogue, manager, fixed)
                if reference_report and fixed:
                    reference_report["coordinate_info"] = {
                        "system": "xyxy on warped current-image canvas",
                        "working_size": [selected_image.rgb.shape[1], selected_image.rgb.shape[0]],
                        "reference_original_size": [ref.original.shape[1], ref.original.shape[0]],
                        "reference_working_to_current_homography": alignment["homography"],
                        "original_reference_polygons": to_reference_original(
                            reference_report["corrected_result"], alignment["homography"],
                            (ref.rgb.shape[1], ref.rgb.shape[0]),
                            (ref.original.shape[1], ref.original.shape[0]))}
                if reference_report:
                    st.download_button("Download independent reference JSON", to_json(reference_report), "reference_audit.json", "application/json")
                if reference_report and report and ready_to_compare:
                    current_names = {r["name"] for r in report["regions"]}
                    reference_names = {r["name"] for r in reference_report["regions"]}
                    if current_names != reference_names:
                        st.warning("Region names must match exactly before comparison.")
                    else:
                        findings = compare_audits(report["analysis"], reference_report["analysis"])
                        comparison = {"status": "Compared — manual verification required", "method": choice,
                                      "user_confirmed_correspondence": True,
                                      "alignment": {k: st.session_state[key][k] for k in ["success", "reason", "metrics", "homography"]}
                                      if fixed else None,
                                      "reference_report": reference_report, "findings": findings}
                        st.dataframe(findings, use_container_width=True, hide_index=True)
                elif reference_report:
                    comparison = {"status": "Independent audit only; no comparison accepted", "reference_report": reference_report}
            except (ValueError, RuntimeError) as error:
                st.error(f"Reference comparison unavailable: {error}")
        else:
            st.info("Upload both current and reference images to audit and compare them.")
    with tabs[3]:
        if report:
            report["reference_comparison"] = comparison
            st.subheader("Export this audit")
            st.caption("Exports contain model settings, original results, committed corrections, catalogue snapshot and comparison evidence.")
            a, b = st.columns(2)
            a.download_button("Download complete JSON", to_json(report), "shelfsense_audit.json", "application/json", use_container_width=True)
            b.download_button("Download evidence CSV", to_csv(report), "shelfsense_audit.csv", "text/csv", use_container_width=True)
            summary = [{k: v for k, v in r.items() if k != "potential_gaps"} for r in report["analysis"]["regions"]]
            st.download_button("Download region summary CSV", pd.DataFrame(summary).to_csv(index=False), "shelf_regions.csv", "text/csv")
            with st.expander("Full audit evidence"):
                st.json(report)
        else:
            st.info("Complete a current shelf audit to enable exports.")


if __name__ == "__main__":
    main()
