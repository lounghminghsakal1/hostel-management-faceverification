import numpy as np
from typing import Any, Dict, Optional, Tuple
from app.face_model import get_face_model, FaceModel

# Recommended ArcFace default threshold for 1:1 verification.
# Typically, cosine similarity for the same identity is > 0.60 (often 0.70 - 0.95),
# whereas different identities typically score < 0.35.
# A threshold of 0.55 provides a balanced trade-off between FAR and FRR.
DEFAULT_VERIFICATION_THRESHOLD: float = 0.50

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

def compute_euclidean_distance(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Computes Euclidean distance between two unit-normalized vectors:
    d^2 = 2 - 2 * cosine_similarity
    For unit vectors, Euclidean distance ranges between 0.0 (identical) and 2.0 (opposite).
    """
    dist = float(np.linalg.norm(embedding_a - embedding_b))
    return round(dist, 4)

def verify_face(
    base_image: Any,
    capture_image: Any,
    threshold: float = DEFAULT_VERIFICATION_THRESHOLD
) -> Dict[str, Any]:
    """
    Compares a registered Base Photo and a Current Captured Photo to verify if
    both belong to the same resident.

    Parameters:
        base_image: Image path (str), raw image bytes (bytes), or numpy array (BGR).
        capture_image: Image path (str), raw image bytes (bytes), or numpy array (BGR).
        threshold: Cosine similarity cutoff (default 0.55).

    Returns:
        Structured dictionary:
        {
            'verified': bool,
            'similarity_score': float,
            'threshold': float,
            'face_detected': {
                'base_image': bool,
                'capture_image': bool
            },
            'message': str
        }
    """
    model: FaceModel = get_face_model()

    # Step 1: Decode and validate base image
    try:
        base_img_bgr = model.decode_image(base_image)
    except Exception as e:
        return {
            "verified": False,
            "similarity_score": 0.0,
            "threshold": threshold,
            "face_detected": {"base_image": False, "capture_image": False},
            "message": f"Failed to load Base Image: {str(e)}"
        }

    # Step 2: Decode and validate capture image
    try:
        capture_img_bgr = model.decode_image(capture_image)
    except Exception as e:
        return {
            "verified": False,
            "similarity_score": 0.0,
            "threshold": threshold,
            "face_detected": {"base_image": False, "capture_image": False},
            "message": f"Failed to load Captured Image: {str(e)}"
        }

    # Step 3: Detect faces in Base Image
    base_det = model.detect_face(base_img_bgr)
    if base_det["status"] != "SUCCESS":
        return {
            "verified": False,
            "similarity_score": 0.0,
            "threshold": threshold,
            "face_detected": {
                "base_image": (base_det["count"] > 0),
                "capture_image": False
            },
            "message": f"Base image issue: {base_det['message']}"
        }

    # Step 4: Detect faces in Captured Image
    cap_det = model.detect_face(capture_img_bgr)
    if cap_det["status"] != "SUCCESS":
        return {
            "verified": False,
            "similarity_score": 0.0,
            "threshold": threshold,
            "face_detected": {
                "base_image": True,
                "capture_image": (cap_det["count"] > 0)
            },
            "message": f"Captured image issue: {cap_det['message']}"
        }

    # Step 5: Align faces & generate 512-dimensional ArcFace embeddings
    try:
        aligned_base = model.align_and_crop_face(base_img_bgr, base_det["landmarks"])
        emb_base = model.generate_embedding(aligned_base)

        aligned_cap = model.align_and_crop_face(capture_img_bgr, cap_det["landmarks"])
        emb_cap = model.generate_embedding(aligned_cap)
    except Exception as e:
        return {
            "verified": False,
            "similarity_score": 0.0,
            "threshold": threshold,
            "face_detected": {"base_image": True, "capture_image": True},
            "message": f"Feature extraction error: {str(e)}"
        }

    # Step 6: Calculate similarity comparison
    similarity_score = compute_cosine_similarity(emb_base, emb_cap)

    # Step 7: Apply configurable verification threshold
    is_verified = bool(similarity_score >= threshold)

    if is_verified:
        message = "Face verified successfully"
    else:
        message = "Face verification failed: low similarity score"

    # Step 8: Return structured result
    return {
        "verified": is_verified,
        "similarity_score": similarity_score,
        "threshold": threshold,
        "face_detected": {
            "base_image": True,
            "capture_image": True
        },
        "message": message
    }
