import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from app.face_model import get_face_model, FaceModel
from app import face_quality

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
# ArcFace cosine similarity is NOT a percentage. For the InsightFace w600k models,
# photos of the same person usually score 0.35-0.85 and different people score
# below ~0.20 (InsightFace's own guidance: < 0.20 "not the same person",
# >= 0.28 "same person"). Showing cosine x 100 made genuine matches look like
# 40-60%, under the app's 65% "trusted" line.
#
# So the cosine is mapped onto the app's bands, piecewise-linearly:
#   cosine <= 0.05 ->   0%
#   cosine    0.20 ->  30%   (below: red, likely someone else)
#   cosine    0.32 ->  65%   (below: yellow, check the photos; from here: green, trusted)
#   cosine >= 0.80 -> 100%   (only near-identical photos; typical good matches land at 75-95%)
# Tune these points if your own data shows a different spread.
CALIBRATION_COSINE = (0.05, 0.20, 0.32, 0.80)
CALIBRATION_PERCENT = (0.0, 30.0, 65.0, 100.0)

# Cosine similarity needed for "verified" (the 65% line above)
DEFAULT_VERIFICATION_THRESHOLD: float = 0.32

# Recent check-in selfies that matched the registered photo well can help when
# a student's look has changed since the photo was taken (beard or moustache
# grown or shaved, haircut, glasses). They can raise the score by at most this
# much cosine above the registered-photo score, so the registered photo always
# anchors the result: an impostor scores low against both.
REFERENCE_BOOST_CAP: float = 0.15
MAX_REFERENCE_IMAGES: int = 3

# Shared pool for downloading images in parallel
_download_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="face-download")


def compute_cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Computes the Cosine Similarity between two 512-dimensional embedding vectors:
    Similarity = (A . B) / (||A|| * ||B||)
    Since both embeddings are already L2-normalized (||A|| = 1, ||B|| = 1),
    this simplifies to the dot product, constrained to [-1.0, 1.0].
    """
    dot_product = float(np.dot(embedding_a, embedding_b))
    # Clip between -1.0 and 1.0 to prevent floating-point anomalies
    similarity = max(-1.0, min(1.0, dot_product))
    return round(similarity, 4)


def compute_face_matching_percentage(similarity_score: float) -> float:
    """
    Converts a cosine similarity score [-1.0, 1.0] into the app's match
    percentage [0, 100] using the calibration points above
    (65%+ trusted, 30-65% check, below 30% likely someone else).
    """
    percent = float(np.interp(similarity_score, CALIBRATION_COSINE, CALIBRATION_PERCENT))
    return round(percent, 2)


def compute_euclidean_distance(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Computes Euclidean distance between two unit-normalized vectors:
    d^2 = 2 - 2 * cosine_similarity
    For unit vectors, Euclidean distance ranges between 0.0 (identical) and 2.0 (opposite).
    """
    dist = float(np.linalg.norm(embedding_a - embedding_b))
    return round(dist, 4)


def _failure(status: str, message: str, threshold: float, base_face: bool = False, capture_face: bool = False,
             issues: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    return {
        "issues": issues or [],
        "verified": False,
        "status": status,
        "similarity_score": 0.0,
        "base_similarity": 0.0,
        "reference_similarity": None,
        "matched_with": None,
        "references_used": 0,
        "face_matching_percentage": 0.0,
        "threshold": threshold,
        "face_detected": {"base_image": base_face, "capture_image": capture_face},
        "message": message,
    }


def _try_decode(model: FaceModel, image: Any):
    try:
        return model.decode_image(image), None
    except Exception as e:  # noqa: BLE001 - reported back to the caller
        return None, str(e)


def _embedding_of(model: FaceModel, image_bgr: np.ndarray, detection: Dict[str, Any]) -> np.ndarray:
    aligned = model.align_and_crop_face(image_bgr, detection["landmarks"])
    return model.generate_embedding(aligned)


def verify_face(
    base_image: Any,
    capture_image: Any,
    threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
    reference_images: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """
    Compares a registered Base Photo and a Current Captured Photo to verify if
    both belong to the same resident.

    Parameters:
        base_image: Image URL/path (str), raw image bytes (bytes), or numpy array (BGR).
        capture_image: Image URL/path (str), raw image bytes (bytes), or numpy array (BGR).
        threshold: Cosine similarity cutoff for `verified` (default 0.32, the 65% line).
        reference_images: Optional recent check-in selfies of the same student that
            already matched the base photo well. Used only to soften changes in look
            (see REFERENCE_BOOST_CAP). Up to MAX_REFERENCE_IMAGES are used; any that
            fail to load or show no clear face are skipped.

    Returns a dict with:
        verified, status, similarity_score (final cosine), base_similarity,
        reference_similarity, matched_with ("base_photo" | "recent_check_ins"),
        references_used, face_matching_percentage, threshold, face_detected, message.

    status is one of:
        VERIFIED, NOT_VERIFIED,
        BASE_IMAGE_UNREADABLE, NO_FACE_IN_BASE, MULTIPLE_FACES_IN_BASE,
        CAPTURED_IMAGE_UNREADABLE, NO_FACE_IN_CAPTURE, MULTIPLE_FACES_IN_CAPTURE,
        LOW_QUALITY_CAPTURE (message says what to fix; details in `issues`),
        PROCESSING_ERROR
    """
    model: FaceModel = get_face_model()
    references = list(reference_images or [])[:MAX_REFERENCE_IMAGES]

    # Step 1: Load all images in parallel (they are usually S3 downloads)
    base_job = _download_pool.submit(_try_decode, model, base_image)
    capture_job = _download_pool.submit(_try_decode, model, capture_image)
    reference_jobs = [_download_pool.submit(_try_decode, model, ref) for ref in references]

    base_img_bgr, base_error = base_job.result()
    if base_img_bgr is None:
        return _failure("BASE_IMAGE_UNREADABLE", f"Failed to load Base Image: {base_error}", threshold)

    capture_img_bgr, capture_error = capture_job.result()
    if capture_img_bgr is None:
        return _failure("CAPTURED_IMAGE_UNREADABLE", f"Failed to load Captured Image: {capture_error}", threshold)

    # Step 2: Detect faces in Base Image (a sideways photo is turned upright)
    base_img_bgr, base_det, _ = model.detect_face_upright(base_img_bgr)
    if base_det["status"] != "SUCCESS":
        status = "MULTIPLE_FACES_IN_BASE" if base_det["status"] == "MULTIPLE_FACES_DETECTED" else "NO_FACE_IN_BASE"
        return _failure(status, f"Base image issue: {base_det['message']}", threshold, base_face=base_det["count"] > 0)

    # Step 3: Detect faces in Captured Image
    capture_img_bgr, cap_det, _ = model.detect_face_upright(capture_img_bgr)
    if cap_det["status"] != "SUCCESS":
        status = "MULTIPLE_FACES_IN_CAPTURE" if cap_det["status"] == "MULTIPLE_FACES_DETECTED" else "NO_FACE_IN_CAPTURE"
        return _failure(status, f"Captured image issue: {cap_det['message']}", threshold, base_face=True, capture_face=cap_det["count"] > 0)

    # Step 3b: A selfie too dark, blurry, far away or turned away gives an unreliable
    # score, so ask for a retake instead (lenient: only extreme cases)
    capture_issues = [i for i in face_quality.assess(capture_img_bgr, cap_det, "capture")["issues"] if i["severity"] == "error"]
    if capture_issues:
        return _failure("LOW_QUALITY_CAPTURE", capture_issues[0]["message"], threshold, base_face=True, capture_face=True, issues=capture_issues)

    # Step 4: Align faces & generate 512-dimensional ArcFace embeddings
    try:
        emb_base = _embedding_of(model, base_img_bgr, base_det)
        emb_cap = _embedding_of(model, capture_img_bgr, cap_det)
    except Exception as e:  # noqa: BLE001
        return _failure("PROCESSING_ERROR", f"Feature extraction error: {str(e)}", threshold, base_face=True, capture_face=True)

    # Step 5: Compare with the registered photo
    base_similarity = compute_cosine_similarity(emb_base, emb_cap)

    # Step 6: Compare with recent verified selfies, if any
    reference_similarities: List[float] = []
    for job in reference_jobs:
        ref_bgr, _ = job.result()
        if ref_bgr is None:
            continue
        ref_bgr, ref_det, _ = model.detect_face_upright(ref_bgr)
        if ref_det["status"] != "SUCCESS":
            continue
        try:
            reference_similarities.append(compute_cosine_similarity(_embedding_of(model, ref_bgr, ref_det), emb_cap))
        except Exception:  # noqa: BLE001 - a bad reference is simply skipped
            continue

    reference_similarity = max(reference_similarities) if reference_similarities else None
    similarity_score = base_similarity
    matched_with = "base_photo"
    if reference_similarity is not None:
        boosted = min(reference_similarity, base_similarity + REFERENCE_BOOST_CAP)
        if boosted > base_similarity:
            similarity_score = round(boosted, 4)
            matched_with = "recent_check_ins"

    face_matching_percentage = compute_face_matching_percentage(similarity_score)

    # Step 7: Apply configurable verification threshold
    is_verified = bool(similarity_score >= threshold)
    message = "Face verified successfully" if is_verified else "Face verification failed: low similarity score"

    # Step 8: Return structured result
    return {
        "issues": [],
        "verified": is_verified,
        "status": "VERIFIED" if is_verified else "NOT_VERIFIED",
        "similarity_score": similarity_score,
        "base_similarity": base_similarity,
        "reference_similarity": reference_similarity,
        "matched_with": matched_with,
        "references_used": len(reference_similarities),
        "face_matching_percentage": face_matching_percentage,
        "threshold": threshold,
        "face_detected": {
            "base_image": True,
            "capture_image": True
        },
        "message": message
    }


def check_face_photo(image: Any, purpose: str = "registration") -> Dict[str, Any]:
    """
    Checks one photo before it's used: exactly one clear, sharp, well-lit,
    front-facing face, big enough and inside the frame.

    purpose: "registration" (strict, for the student's photo on file) or
             "capture" (lenient, the same rules check-in selfies get).

    Returns {ok, status, message, issues, metrics, purpose}:
        ok      - False when any issue is an error (the photo shouldn't be used)
        status  - OK, OK_WITH_WARNINGS or REJECTED
        issues  - [{code, severity: "error" | "warning", message}], most important first
        metrics - the measured values (face size, pose, sharpness, brightness, ...)
    """
    model: FaceModel = get_face_model()
    image_bgr, error = _try_decode(model, image)
    if image_bgr is None:
        issues = [face_quality.issue("IMAGE_UNREADABLE")]
        return {**face_quality.summarize(issues), "issues": issues, "metrics": {"error": error}, "purpose": purpose}

    upright, detection, rotation = model.detect_face_upright(image_bgr)
    metrics: Dict[str, Any] = {"image_width": image_bgr.shape[1], "image_height": image_bgr.shape[0], "rotation": rotation}

    if detection["status"] == "NO_FACE_DETECTED":
        issues = [face_quality.issue("NO_FACE")]
        if min(image_bgr.shape[:2]) < face_quality.THRESHOLDS[purpose].min_image_side:
            issues.append(face_quality.issue("LOW_RESOLUTION"))
    elif detection["status"] == "MULTIPLE_FACES_DETECTED":
        issues = [face_quality.issue("MULTIPLE_FACES")]
        metrics["face_count"] = detection["count"]
    else:
        quality = face_quality.assess(upright, detection, purpose)
        issues = quality["issues"]
        metrics.update(quality["metrics"])
        if rotation and purpose == "registration":
            issues.insert(0, face_quality.issue("PHOTO_SIDEWAYS"))

    # errors first, then warnings
    issues.sort(key=lambda i: 0 if i["severity"] == "error" else 1)
    return {**face_quality.summarize(issues), "issues": issues, "metrics": metrics, "purpose": purpose}
