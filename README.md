# Hostel Management � AI Face Verification System

A high-performance, Python-based **1:1 Face Verification System** designed specifically for hostel gate access and student attendance control.

This system compares a registered **Base Photo** against a live **Captured Photo** and verifies whether both images belong to the same student using deep convolutional neural network (CNN) face embeddings.

---

## Key Principles & Objective

- **Verification (1:1), NOT Classification (1:N)**:
  Does **not** classify across a fixed database of faces (e.g. "Person A: 95%"). Instead, it generates invariant 512-dimensional biometric feature embeddings and compares them using cosine similarity.
- **Robust Invariance**:
  Robust against normal real-world variations including glasses vs. no glasses, beards/moustaches, haircuts, facial expressions, minor angles ($\pm 30^\circ$), and varying lighting conditions.
- **No Manual Feature Engineering**:
  Uses complete holistic facial representation via ArcFace (Additive Angular Margin Loss), avoiding error-prone manual rules for eyes, ears, or lips.

---

## Project Structure

```text
hostel_face_verification/
�
+-- app/
�   +-- __init__.py
�   +-- main.py             # FastAPI web application & POST /face/verify endpoint
�   +-- face_model.py       # ArcFace ONNX model loader & YuNet face detector
�   +-- face_service.py     # verify_face() logic, similarity & threshold evaluation
�
+-- models/
�   +-- download_models.py  # Automatic downloader for ONNX models
�   +-- arcface_model.onnx  # Pretrained ArcFace 512-d feature extractor (13.6 MB)
�   +-- face_detection_yunet.onnx # OpenCV YuNet CNN face detector (232 KB)
�
+-- test_images/
�   +-- base/               # Test base registered photos
�   +-- capture/            # Test captured variations & negative controls
�   +-- setup_test_images.py# Downloads benchmark evaluation images
�
+-- test_verification.py    # Local test suite for Phase 8 (FAR/FRR evaluation)
+-- test_api.py             # API endpoint automated test
+-- requirements.txt        # Pinned Python dependencies
+-- README.md               # Documentation & integration guide
```

---

## How to Run This Project (Step by Step)

Step 1: Open a terminal (PowerShell) and navigate to the project folder. Command: cd "d:\Backend Projects\hostel-management-faceverification"

Step 2: Delete the old, broken virtual environment that came with the repo (it was built on a different computer). Command: Remove-Item -Recurse -Force venv

Step 3: Create a fresh virtual environment. Command: python -m venv venv

Step 4: Activate the virtual environment. Command: .\venv\Scripts\Activate.ps1
You should see (venv) appear at the start of your terminal prompt once it is active.

Step 5: Install the required dependencies. Command: pip install -r requirements.txt

Step 6: Confirm the AI models are present. The models folder should already contain arcface_model.onnx and face_detection_yunet.onnx, so this step can be skipped. Only if either file is missing, run: python models/download_models.py

Step 7 (optional): Set up benchmark test images, only needed if you plan to run the local verification test suite. Command: python test_images/setup_test_images.py

Step 8: Start the FastAPI server. Command: uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

Step 9: Open your web browser and go to the interactive Swagger docs. Address: http://127.0.0.1:8000/docs
From there you can upload a base photo and a captured photo directly through the UI and test live verification.

Step 10 (optional): Instead of the server, you can run the automated test suites. For the local FAR/FRR evaluation, command: python test_verification.py
To test a running API endpoint, command: python test_api.py or python test_phase4_5.py

---

## API Documentation

### `POST /face/check` (JSON): is this photo good enough?

Used by the backend when a hostel admin uploads a student's photo (`POST /students/image_check`), and again when the student is saved. A poor photo on file makes every later check-in score low, so it's checked strictly.

#### Request body
| Field | Type | Required | Description |
|---|---|---|---|
| `image` | string (URL) | Yes | The photo to check |
| `purpose` | `"registration"` \| `"capture"` | No (default `registration`) | `registration`: strict, for a student photo on file. `capture`: the lenient rules check-in selfies get. |

#### Response (`200 OK`)
```json
{
  "ok": false,
  "status": "REJECTED",
  "message": "The face is turned to the side. Ask the student to look straight at the camera.",
  "issues": [
    { "code": "NOT_FACING_CAMERA", "severity": "error", "message": "The face is turned to the side. Ask the student to look straight at the camera." }
  ],
  "metrics": { "detection_confidence": 0.909, "inter_eye_px": 57.2, "yaw": 0.253, "pitch": 0.53, "roll": 6.7, "sharpness": 179.4, "brightness": 141.0, "...": "..." },
  "purpose": "registration"
}
```

`status` is `OK`, `OK_WITH_WARNINGS` (usable) or `REJECTED` (`ok: false`). Issues, errors first:

| Code | Severity | Means |
|---|---|---|
| `IMAGE_UNREADABLE` | error | Not a JPG/PNG, or the download failed |
| `NO_FACE` | error | No face found |
| `MULTIPLE_FACES` | error | Another face at least half the size of the largest |
| `PHOTO_SIDEWAYS` | error | The face is only upright when the photo is turned |
| `LOW_RESOLUTION` | error | Shorter side under 300 px |
| `FACE_TOO_SMALL` | error | Eyes under 55 px apart, or face under 18% of the photo width |
| `FACE_CUT_OFF` | error | Part of the face is outside the photo |
| `FACE_NOT_CLEAR` | error | The detector isn't confident it's a face (< 0.80) |
| `NOT_FACING_CAMERA` | error | Head turned sideways |
| `LOOKING_UP_OR_DOWN` | error | Head tilted up or down |
| `BLURRY` | error | Out of focus (sharpness < 20) |
| `TOO_DARK` / `TOO_BRIGHT` | error | Face brightness under 45 / over 225 (of 255) |
| `SLIGHTLY_BLURRY` | warning | Sharpness 20-40 |
| `HEAD_TILTED` | warning | Eye line tilted more than 15 degrees |
| `LOW_CONTRAST` | warning | Washed out |
| `UNEVEN_LIGHTING` | warning | One side of the face in shadow |
| `FACE_NOT_CENTERED` | warning | Face far from the middle |

The thresholds live in `app/face_quality.py` (`REGISTRATION`, `CAPTURE`). They were calibrated on the test photos and degraded copies of them (blurred, dark, overexposed, tiny, sideways).

### `POST /face/verify` (JSON)

The Node backend calls this with signed S3 URLs. Bind the server to `127.0.0.1` so only the backend can reach it.

#### Request body
| Field | Type | Required | Description |
|---|---|---|---|
| `base_image` | string (URL) | Yes | The student's registered photo |
| `captured_image` | string (URL) | Yes | Tonight's check-in selfie |
| `reference_images` | string[] (URLs) | No | Recent check-in selfies that matched the registered photo well (the backend sends up to 3 from the last 30 days with 80%+). Helps when a beard, moustache or haircut has changed. |
| `threshold` | float | No (default `0.32`) | Cosine similarity needed for `verified` (the 65% line) |

#### Response (`200 OK`)
```json
{
  "verified": true,
  "status": "VERIFIED",
  "similarity_score": 0.703,
  "base_similarity": 0.703,
  "reference_similarity": null,
  "matched_with": "base_photo",
  "references_used": 0,
  "face_matching_percentage": 100.0,
  "threshold": 0.32,
  "face_detected": { "base_image": true, "capture_image": true },
  "message": "Face verified successfully"
}
```

`status` is one of:

| Status | Meaning | What the backend does |
|---|---|---|
| `VERIFIED` / `NOT_VERIFIED` | Compared successfully | Saves the record with `face_matching_percentage` |
| `NO_FACE_IN_CAPTURE`, `CAPTURED_IMAGE_UNREADABLE` | Selfie has no clear face | 422, the student retakes the selfie (nothing saved) |
| `LOW_QUALITY_CAPTURE` | Selfie far too dark, blurry, small or turned away to score reliably (lenient `capture` rules; `message` says what to fix) | 422 with that message, retake |
| `MULTIPLE_FACES_IN_CAPTURE` | Another person is clearly in the selfie | 422, retake with only the student in frame |
| `NO_FACE_IN_BASE`, `MULTIPLE_FACES_IN_BASE`, `BASE_IMAGE_UNREADABLE` | The registered photo is unusable | 409, ask the hostel office for a new photo |
| `PROCESSING_ERROR` | Unexpected model error | 503 |

A small face far behind the student (or a poster) is ignored: the largest face is used. "Multiple faces" means a second face at least half the size of the largest.

### How the percentage is calculated

1. **Detect** the largest face with YuNet (on a copy at most 1024 px, for speed). If the face isn't clearly upright (confidence under 0.93), the photo is also tried turned 90/180/270 degrees and the most confident orientation is used. YuNet "finds" faces in rotated photos with wrong landmarks, which used to score a genuine sideways selfie at 0%.
2. **Align** it to 112×112 with a least-squares similarity transform over all 5 landmarks (eyes, nose, mouth corners), as InsightFace does.
3. **Embed** with ArcFace (`w600k_mbf`) in **RGB** order, averaging the face and its mirror image ("flip test"). The embedding is L2-normalised.
4. **Compare** with cosine similarity against the registered photo (`base_similarity`).
5. **Recent check-ins** (optional): the best similarity against the reference selfies can raise the score by at most **0.15** above `base_similarity`. The registered photo always anchors the result, so an impostor stays low even if one of their own selfies were used as a reference.
6. **Calibrate** the cosine into the app's bands (cosine is not a percentage):

| Cosine | Percentage | App colour |
|---|---|---|
| ≤ 0.05 | 0% | red |
| 0.20 | 30% | below: red (likely someone else) |
| 0.32 | 65% | below: yellow (check the photos); from here: green (trusted) |
| ≥ 0.80 | 100% | green (only near-identical photos; good matches land at 75-95%) |

The points are in `app/face_service.py` (`CALIBRATION_COSINE`, `CALIBRATION_PERCENT`). They follow InsightFace's guidance for these models (under 0.20 not the same person, 0.28+ the same person). Re-tune them once you have your own data (see below).

**Why it changed (v1.1):** before, the percentage was cosine × 100, so genuine matches typically showed 40–70% (below the 65% line). The model was also fed BGR instead of RGB, and RANSAC alignment could ignore a landmark. On the test photos the same person went from cosine 0.59 to 0.70–0.73 with the fixes, while different people stayed near 0. The endpoint now runs in a worker thread, so many students checking in at once don't queue behind each other.

### Tests

```bash
python test_verification.py   # the original 6 scenarios
python test_scoring.py        # calibration, reference cap, impostor checks
python test_quality.py        # photo checks, selfie retakes, sideways photos
```

---

## Metric Evaluation: FAR, FRR & Threshold Tuning

1. **False Accept Rate (FAR)**:
   $$\text{FAR} = \frac{\text{False Positives}}{\text{Total Impostor Attempts}}$$
   Proportion of unauthorized non-residents who are mistakenly verified. In a hostel setting, high FAR creates a security risk.

2. **False Reject Rate (FRR)**:
   $$\text{FRR} = \frac{\text{False Negatives}}{\text{Total Genuine Attempts}}$$
   Proportion of actual hostel residents who are incorrectly rejected. High FRR causes gate delays and inconvenience.

3. **Tuning the calibration**:
   - Collect 50-100 genuine pairs (a student's photo vs their own check-ins) and 50-100 impostor pairs.
   - Look at the cosine values (`base_similarity` in the response) for both groups.
   - Put the 30% point where impostors almost never reach, and the 65% point where genuine pairs almost always clear it.
   - Update `CALIBRATION_COSINE` and `DEFAULT_VERIFICATION_THRESHOLD` (the 65% cosine) in `app/face_service.py`.

---

## Security & Biometric Privacy Notes

- **Transient Processing**: Never store raw captured live frames permanently; process in memory and discard.
- **Embedding Storage**: When adding database persistence later, only store the mathematical 512-dimensional vector embedding, not the resident's raw biometric images.
- **HTTPS Enforcement**: Always deploy with TLS/HTTPS in production to prevent biometric intercept.
