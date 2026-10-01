"""Exact rectangle unions and conservative shelf evidence summaries."""
from collections import Counter


def intersection(a: list[float], b: list[float]) -> list[float] | None:
    box = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return box if box[2] > box[0] and box[3] > box[1] else None


def merged_intervals(intervals: list[tuple]) -> list[list[float]]:
    merged = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(right, merged[-1][1])
        else:
            merged.append([left, right])
    return merged


def union_area(boxes: list[list[float]]) -> float:
    """Sweep along x; sum merged y intervals in each slab."""
    edges = sorted({x for box in boxes for x in (box[0], box[2])})
    area = 0.0
    for left, right in zip(edges, edges[1:]):
        intervals = [(b[1], b[3]) for b in boxes if b[0] < right and b[2] > left]
        area += (right - left) * sum(y2 - y1 for y1, y2 in merged_intervals(intervals))
    return area


def assign_region(box: list[float], regions: list[dict]) -> str | None:
    """Assign by box centre, once; right/bottom boundary belongs to next region."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for region in regions:
        x1, y1, x2, y2 = region["box"]
        if x1 <= cx < x2 and y1 <= cy < y2:
            return region["name"]
    return None


def potential_gaps(region: list[float], boxes: list[list[float]],
                   minimum_fraction: float) -> list[dict]:
    if not 0 < minimum_fraction <= 1:
        raise ValueError("Minimum gap fraction must be in (0, 1].")
    clipped = [cut for b in boxes if (cut := intersection(b, region))]
    intervals = merged_intervals([(b[0], b[2]) for b in clipped])
    x1, y1, x2, y2 = region
    cursor, gaps = x1, []
    for left, right in intervals + [[x2, x2]]:
        if left - cursor >= minimum_fraction * (x2 - x1):
            gaps.append({"box": [cursor, y1, left, y2],
                         "label": "Potential gap — verify",
                         "evidence": "No annotated/detected box covers this horizontal interval."})
        cursor = max(cursor, right)
    return gaps


def analyze_shelf(detections: list[dict], regions: list[dict], catalogue: list[dict],
                  gap_fraction: float = 0.08) -> dict:
    assigned = {d["id"]: assign_region(d["box"], regions) for d in detections}
    scoped = [d for d in detections if assigned[d["id"]] is not None]
    counts = Counter(d["product_id"] for d in scoped if d.get("product_id"))
    rows = []
    for region in regions:
        name, box = region["name"], region["box"]
        inside = [d for d in scoped if assigned[d["id"]] == name]
        # All intersections contribute visual coverage, even a boundary-straddling box.
        clipped = [cut for d in detections if (cut := intersection(d["box"], box))]
        area = (box[2] - box[0]) * (box[3] - box[1])
        rows.append({"region": name, "visible_facings": len(inside),
                     "estimated_visual_occupancy": union_area(clipped) / area,
                     "counts_by_product": dict(Counter(d["product_id"] for d in inside
                                                        if d.get("product_id"))),
                     "unknown_count": sum(not d.get("product_id") for d in inside),
                     "potential_gaps": potential_gaps(box, [d["box"] for d in detections], gap_fraction)})
    targets, misplaced = [], []
    names = {r["name"] for r in regions}
    for product in catalogue:
        pid, expected = product["product_id"], product.get("expected_region")
        if expected:
            for detection in scoped:
                actual = assigned[detection["id"]]
                if detection.get("product_id") == pid and actual != expected:
                    misplaced.append({"product_id": pid, "box_id": detection["id"],
                                      "expected_region": expected, "observed_region": actual,
                                      "label": "Possible misplaced product — verify"})
        target = product.get("minimum_facings")
        if target is not None:
            assessed = not expected or expected in names
            observed = (sum(d.get("product_id") == pid and assigned[d["id"]] == expected
                            for d in scoped) if expected else counts.get(pid, 0))
            targets.append({"product_id": pid, "region": expected or "all audited regions",
                            "target": target, "observed_visible_facings": observed if assessed else None,
                            "display_facing_shortfall": max(0, target - observed) if assessed else None,
                            "status": "Verify identities and missed boxes" if assessed else "Expected region not audited"})
    return {"visible_facings": len(scoped), "outside_region_count": len(detections) - len(scoped),
            "counts_by_product": dict(counts),
            "unknown_product_count": sum(not d.get("product_id") for d in scoped),
            "needs_verification_count": sum(d.get("status") == "Needs verification" for d in scoped),
            "regions": rows, "targets": targets, "possible_misplacements": misplaced}


def compare_audits(current: dict, reference: dict) -> list[dict]:
    """Compare only corresponding audited regions; no pixel-based stock claims."""
    previous = {r["region"]: r for r in reference["regions"]}
    findings = []
    for row in current["regions"]:
        old = previous.get(row["region"])
        if old is None:
            continue
        products = set(old["counts_by_product"]) | set(row["counts_by_product"])
        changes = {pid: row["counts_by_product"].get(pid, 0) - old["counts_by_product"].get(pid, 0)
                   for pid in sorted(products)}
        findings.append({"region": row["region"],
                         "visible_facing_delta": row["visible_facings"] - old["visible_facings"],
                         "occupancy_delta": row["estimated_visual_occupancy"] - old["estimated_visual_occupancy"],
                         "product_facing_deltas": changes,
                         "possible_missing_facings": {pid: -n for pid, n in changes.items() if n < 0},
                         "evidence": "Counts from separately audited corresponding regions; verify lower counts, identities and occlusion. No inventory or sales inference."})
    return findings
