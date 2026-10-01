"""Conservative ORB/RANSAC registration; alignment is not evidence of stock."""
import cv2
import numpy as np


def align_reference(reference: np.ndarray, current: np.ndarray) -> dict:
    def reject(reason: str, **metrics) -> dict:
        return {"success": False, "reason": reason, "metrics": metrics,
                "warped": None, "homography": None}

    orb = cv2.ORB_create(nfeatures=4000)
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(current, cv2.COLOR_RGB2GRAY), None)
    if des1 is None or des2 is None:
        return reject("Not enough texture for automatic alignment.")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]
    # Repeated packaging can send many source keypoints to one target keypoint.
    unique = {}
    for match in sorted(good, key=lambda m: m.distance):
        unique.setdefault(match.trainIdx, match)
    good = list(unique.values())
    if len(good) < 16:
        return reject("Fewer than 16 distinct reliable feature matches.", matches=len(good))
    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None or mask is None or not np.all(np.isfinite(matrix)):
        return reject("Homography estimation failed.")
    inliers = mask.ravel().astype(bool)
    count, ratio = int(inliers.sum()), float(inliers.mean())
    if count < 14 or ratio < 0.55:
        return reject("Insufficient geometric agreement.", inliers=count, inlier_ratio=ratio)
    rh, rw = reference.shape[:2]
    h, w = current.shape[:2]
    coverage_src = cv2.contourArea(cv2.convexHull(src[inliers])) / (rh * rw)
    coverage_dst = cv2.contourArea(cv2.convexHull(dst[inliers])) / (h * w)
    projected = cv2.perspectiveTransform(src[inliers, None, :], matrix)[:, 0]
    error = float(np.median(np.linalg.norm(projected - dst[inliers], axis=1)))
    corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]])
    quad = cv2.perspectiveTransform(corners[None], matrix)[0]
    area_ratio = abs(cv2.contourArea(quad)) / (h * w)
    canvas = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if not cv2.isContourConvex(quad) or cv2.contourArea(quad, oriented=True) <= 0:
        return reject("Warp is folded, reflected or degenerate.")
    overlap, _ = cv2.intersectConvexConvex(quad, canvas)
    overlap_ratio = float(overlap / (h * w))
    metrics = {"inliers": count, "inlier_ratio": ratio, "median_error_px": error,
               "reference_feature_coverage": float(coverage_src),
               "current_feature_coverage": float(coverage_dst), "overlap_ratio": overlap_ratio,
               "warped_area_ratio": area_ratio}
    if min(coverage_src, coverage_dst) < 0.15 or error > 2.5 or not 0.6 <= area_ratio <= 1.6 or overlap_ratio < 0.85:
        return reject("Viewpoint, coverage or reprojection quality is unreliable.", **metrics)
    warp = cv2.warpPerspective(reference, matrix, (w, h))
    valid = cv2.warpPerspective(np.ones((rh, rw), np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST)
    return {"success": True, "reason": "Geometric checks passed; visually confirm correspondence.",
            "metrics": metrics, "warped": warp, "valid_mask": valid,
            "homography": matrix.tolist()}


def regions_covered(regions: list[dict], valid_mask: np.ndarray) -> bool:
    """Reject an audit region extending into black, unobserved warp padding."""
    for region in regions:
        x1, y1 = np.floor(region["box"][:2]).astype(int)
        x2, y2 = np.ceil(region["box"][2:]).astype(int)
        if valid_mask[y1:y2, x1:x2].size == 0 or valid_mask[y1:y2, x1:x2].mean() < 0.995:
            return False
    return True


def to_reference_original(detections: list[dict], homography: list[list[float]],
                          reference_size: tuple[int, int], original_size: tuple[int, int]) -> list[dict]:
    """Inverse-map warped rectangles as quadrilaterals; perspective loses rectangularity."""
    inverse = np.linalg.inv(np.asarray(homography, dtype=np.float64))
    rw, rh = reference_size
    ow, oh = original_size
    result = []
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        corners = np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
        mapped = cv2.perspectiveTransform(corners, inverse)[0]
        mapped *= np.array([ow / rw, oh / rh])
        result.append({"id": detection["id"], "polygon": mapped.tolist()})
    return result
