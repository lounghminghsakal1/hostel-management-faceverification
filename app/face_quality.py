"""
Photo quality checks: is there exactly one clear, sharp, well-lit, front-facing
face, big enough and inside the frame?

Used two ways:
  * "registration": strict, for the student's photo on file (POST /face/check).
    A poor registration photo makes every later check-in score low.
  * "capture": lenient, for a check-in selfie (POST /face/verify), so a selfie
    that is too dark, blurry, far away or turned away gets a "retake" instead
    of a misleading low match score.

All checks use OpenCV + numpy only (no extra models).
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Where the nose sits between the eye line (0) and mouth line (1) on a
# straight-on face, from the ArcFace reference landmarks.
FRONTAL_PITCH_RATIO = 0.49


@dataclass(frozen=True)
class QualityThresholds:
    min_image_side: int            # shorter image side, px
    min_inter_eye_px: float        # distance between the eye centres, px
    min_face_width_ratio: float    # face box width / image width
    min_detection_confidence: float
    max_yaw_ratio: float           # nose offset from the face midline / eye distance
    min_pitch_ratio: float         # looking up
    max_pitch_ratio: float         # looking down
    max_roll_degrees: float        # head tilted sideways
    min_sharpness: float           # Laplacian variance of the face at 160x160 (below: blurry, error)
    soft_sharpness: float          # below this: slightly soft (warning)
    min_brightness: float          # mean grey level of the face (0-255)
    max_brightness: float
    min_contrast: float            # grey level standard deviation of the face
    max_side_light_difference: float  # mean grey difference between face halves
    max_center_offset_x: float     # face centre's horizontal distance from the image centre / width
    center_y_range: tuple          # allowed face centre height / image height (portraits sit high)


# Calibrated on the test photos and degraded copies of them (blur, dark,
# overexposed, tiny, sideways). The registration photo is checked strictly,
# because every later check-in is compared with it.
REGISTRATION = QualityThresholds(
    min_image_side=300,
    min_inter_eye_px=55,
    min_face_width_ratio=0.18,
    min_detection_confidence=0.80,
    max_yaw_ratio=0.22,
    min_pitch_ratio=0.32,
    max_pitch_ratio=0.70,
    max_roll_degrees=15,
    min_sharpness=20,
    soft_sharpness=40,
    min_brightness=45,
    max_brightness=225,
    min_contrast=20,
    max_side_light_difference=55,
    max_center_offset_x=0.25,
    center_y_range=(0.12, 0.72),
)

# ArcFace (with flip averaging) kept a strong match on heavily blurred, dark
# and small selfies in testing, so selfies are only sent back for a retake in
# extreme cases where the score would be unreliable.
CAPTURE = QualityThresholds(
    min_image_side=160,
    min_inter_eye_px=20,
    min_face_width_ratio=0.05,
    min_detection_confidence=0.60,
    max_yaw_ratio=0.45,
    min_pitch_ratio=0.15,
    max_pitch_ratio=0.85,
    max_roll_degrees=45,
    min_sharpness=4,
    soft_sharpness=0,        # no warning for selfies
    min_brightness=20,
    max_brightness=245,
    min_contrast=0,          # not checked for selfies
    max_side_light_difference=999,
    max_center_offset_x=999,
    center_y_range=(-1.0, 2.0),
)

THRESHOLDS = {"registration": REGISTRATION, "capture": CAPTURE}

# User-facing messages. The app shows these as they are, so keep them plain.
MESSAGES = {
    "IMAGE_UNREADABLE": "This file couldn't be opened as a photo. Upload a JPG or PNG.",
    "LOW_RESOLUTION": "The photo is too small. Use a larger, clearer photo.",
    "NO_FACE": "No face found in the photo. Use a clear, front-facing photo of the student.",
    "PHOTO_SIDEWAYS": "The photo is sideways or upside down. Rotate it so the face is upright.",
    "MULTIPLE_FACES": "More than one face is in the photo. Use a photo with only the student.",
    "FACE_NOT_CLEAR": "The face can't be identified clearly. Take a sharper photo in good light, facing the camera.",
    "FACE_TOO_SMALL": "The face is too small or far away. Move closer so the face fills more of the photo.",
    "FACE_CUT_OFF": "Part of the face is outside the photo. Keep the whole face in the frame.",
    "NOT_FACING_CAMERA": "The face is turned to the side. Ask the student to look straight at the camera.",
    "LOOKING_UP_OR_DOWN": "The head is tilted up or down. Ask the student to look straight at the camera.",
    "HEAD_TILTED": "The head is tilted sideways. Keep the head straight.",
    "BLURRY": "The photo is blurry or out of focus. Hold the phone steady and tap the face to focus.",
    "SLIGHTLY_BLURRY": "The photo is a little soft. A sharper photo gives more reliable attendance checks.",
    "TOO_DARK": "The photo is too dark. Take it in better light.",
    "TOO_BRIGHT": "The photo is too bright or washed out. Avoid direct light or flash on the face.",
    "LOW_CONTRAST": "The photo looks washed out. Take it in even, natural light.",
    "UNEVEN_LIGHTING": "One side of the face is in shadow. Face the light evenly.",
    "FACE_NOT_CENTERED": "The face isn't centred. Keep the face in the middle of the photo.",
}

# Problems that make a registration photo unusable. Everything else is a warning.
ERROR_CODES = {
    "IMAGE_UNREADABLE", "LOW_RESOLUTION", "NO_FACE", "PHOTO_SIDEWAYS", "MULTIPLE_FACES",
    "FACE_NOT_CLEAR", "FACE_TOO_SMALL", "FACE_CUT_OFF", "NOT_FACING_CAMERA",
    "LOOKING_UP_OR_DOWN", "BLURRY", "TOO_DARK", "TOO_BRIGHT",
}


def issue(code: str) -> Dict[str, str]:
    return {"code": code, "severity": "error" if code in ERROR_CODES else "warning", "message": MESSAGES[code]}


def _face_crop_gray(image_bgr: np.ndarray, bbox: List[int], size: int = 160) -> Optional[np.ndarray]:
    """The face box (slightly tightened to skip hair/background), grey, resized to size x size."""
    x, y, w, h = bbox
    ih, iw = image_bgr.shape[:2]
    pad_x, pad_y = int(w * 0.10), int(h * 0.08)
    x0, y0 = max(0, x + pad_x), max(0, y + pad_y)
    x1, y1 = min(iw, x + w - pad_x), min(ih, y + h - pad_y)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    gray = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def pose_from_landmarks(landmarks: np.ndarray) -> Dict[str, float]:
    """
    Rough head pose from the 5 landmarks (right eye, left eye, nose, right mouth, left mouth):
      roll  - degrees the eye line is tilted
      yaw   - sideways nose offset from the face midline, in eye distances (0 = straight on)
      pitch - nose position between eye line (0) and mouth line (1); ~0.49 straight on
    """
    re, le, nose, rm, lm = [p.astype(np.float64) for p in landmarks]
    eye_vec = le - re
    inter_eye = float(np.linalg.norm(eye_vec))
    roll = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))

    # Undo the roll so yaw/pitch are measured on an upright face
    angle = -np.radians(roll)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = (re + le) / 2
    re, le, nose, rm, lm = [rot @ (p - center) for p in (re, le, nose, rm, lm)]

    eye_mid = (re + le) / 2
    mouth_mid = (rm + lm) / 2
    midline_x = (eye_mid[0] + mouth_mid[0]) / 2
    yaw = float((nose[0] - midline_x) / inter_eye) if inter_eye > 0 else 0.0
    eye_to_mouth = mouth_mid[1] - eye_mid[1]
    pitch = float((nose[1] - eye_mid[1]) / eye_to_mouth) if eye_to_mouth > 0 else FRONTAL_PITCH_RATIO
    return {"roll": round(roll, 1), "yaw": round(yaw, 3), "pitch": round(pitch, 3), "inter_eye_px": round(inter_eye, 1)}


def assess(image_bgr: np.ndarray, detection: Dict[str, Any], purpose: str = "registration") -> Dict[str, Any]:
    """
    Quality of the chosen face in an image. `detection` is FaceModel.detect_face's
    result (status SUCCESS). Returns {"issues": [...], "metrics": {...}}.
    """
    t = THRESHOLDS[purpose]
    issues: List[Dict[str, str]] = []
    ih, iw = image_bgr.shape[:2]
    x, y, w, h = detection["bbox"]
    pose = pose_from_landmarks(detection["landmarks"])
    gray = _face_crop_gray(image_bgr, detection["bbox"])

    metrics: Dict[str, Any] = {
        "image_width": iw,
        "image_height": ih,
        "face_count": detection.get("count", 1),
        "detection_confidence": round(float(detection.get("confidence", 0.0)), 3),
        "face_width_px": w,
        "face_width_ratio": round(w / iw, 3) if iw else 0,
        **pose,
    }

    if min(ih, iw) < t.min_image_side:
        issues.append(issue("LOW_RESOLUTION"))

    # Face box sticking out of the frame (allow a little: boxes include forehead/chin margin)
    margin = 0.08 * w
    if x < -margin or y < -margin or x + w > iw + margin or y + h > ih + margin:
        issues.append(issue("FACE_CUT_OFF"))

    if pose["inter_eye_px"] < t.min_inter_eye_px or (iw and w / iw < t.min_face_width_ratio):
        issues.append(issue("FACE_TOO_SMALL"))

    if metrics["detection_confidence"] < t.min_detection_confidence:
        issues.append(issue("FACE_NOT_CLEAR"))

    if abs(pose["yaw"]) > t.max_yaw_ratio:
        issues.append(issue("NOT_FACING_CAMERA"))
    if not (t.min_pitch_ratio <= pose["pitch"] <= t.max_pitch_ratio):
        issues.append(issue("LOOKING_UP_OR_DOWN"))
    if abs(pose["roll"]) > t.max_roll_degrees:
        issues.append(issue("HEAD_TILTED"))

    if gray is not None:
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        contrast = float(gray.std())
        half = gray.shape[1] // 2
        side_difference = float(abs(gray[:, :half].mean() - gray[:, half:].mean()))
        metrics.update({
            "sharpness": round(sharpness, 1),
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "side_light_difference": round(side_difference, 1),
        })
        if brightness < t.min_brightness:
            issues.append(issue("TOO_DARK"))
        elif brightness > t.max_brightness:
            issues.append(issue("TOO_BRIGHT"))
        elif contrast < t.min_contrast:
            issues.append(issue("LOW_CONTRAST"))
        # Dark photos also read as "blurry"; report the darkness only
        if brightness >= t.min_brightness:
            if sharpness < t.min_sharpness:
                issues.append(issue("BLURRY"))
            elif sharpness < t.soft_sharpness:
                issues.append(issue("SLIGHTLY_BLURRY"))
        if side_difference > t.max_side_light_difference:
            issues.append(issue("UNEVEN_LIGHTING"))

    if iw and ih:
        offset_x = abs((x + w / 2) / iw - 0.5)
        center_y = (y + h / 2) / ih
        metrics["center_offset_x"] = round(offset_x, 3)
        metrics["center_y"] = round(center_y, 3)
        if offset_x > t.max_center_offset_x or not (t.center_y_range[0] <= center_y <= t.center_y_range[1]):
            issues.append(issue("FACE_NOT_CENTERED"))

    return {"issues": issues, "metrics": metrics}


def summarize(issues: List[Dict[str, str]]) -> Dict[str, Any]:
    errors = [i for i in issues if i["severity"] == "error"]
    if errors:
        return {"ok": False, "status": "REJECTED", "message": errors[0]["message"]}
    if issues:
        return {"ok": True, "status": "OK_WITH_WARNINGS", "message": issues[0]["message"]}
    return {"ok": True, "status": "OK", "message": "The photo looks good."}
