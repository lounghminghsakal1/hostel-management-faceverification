# Hostel Management – AI Face Verification System

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
¦
+-- app/
¦   +-- __init__.py
¦   +-- main.py             # FastAPI web application & POST /face/verify endpoint
¦   +-- face_model.py       # ArcFace ONNX model loader & YuNet face detector
¦   +-- face_service.py     # verify_face() logic, similarity & threshold evaluation
¦
+-- models/
¦   +-- download_models.py  # Automatic downloader for ONNX models
¦   +-- arcface_model.onnx  # Pretrained ArcFace 512-d feature extractor (13.6 MB)
¦   +-- face_detection_yunet.onnx # OpenCV YuNet CNN face detector (232 KB)
¦
+-- test_images/
¦   +-- base/               # Test base registered photos
¦   +-- capture/            # Test captured variations & negative controls
¦   +-- setup_test_images.py# Downloads benchmark evaluation images
¦
+-- test_verification.py    # Local test suite for Phase 8 (FAR/FRR evaluation)
+-- test_api.py             # API endpoint automated test
+-- requirements.txt        # Pinned Python dependencies
+-- README.md               # Documentation & integration guide
```

---

## Exact Installation & Setup Commands

### Step 1: Clone or Navigate to the Project Folder
```powershell
cd C:\Users\ELCOT\.gemini\antigravity\scratch\hostel_face_verification
```

### Step 2: Create and Activate Python Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Download Pretrained Models (YuNet & ArcFace)
```powershell
python models/download_models.py
```

### Step 5: (Optional) Setup Benchmark Test Images
```powershell
python test_images/setup_test_images.py
```

---

## How to Run the System

### 1. Run the Local Test Suite
To test same-person, different-person, glasses, beard, lighting variations, and edge cases:
```powershell
python test_verification.py
```

### 2. Run the FastAPI Server
To launch the backend API:
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Server will start at: `http://127.0.0.1:8000`

### 3. Open Interactive Swagger / OpenAPI Docs
Visit in your web browser:
```text
http://127.0.0.1:8000/docs
```
You can upload photos directly through the UI and test live verification.

---

## API Documentation

### `POST /face/verify`

Accepts both `base_image` and `capture_image` in a **single multipart form request**.

#### Request Parameters (Form Data):
| Field | Type | Required | Description |
|---|---|---|---|
| `base_image` | File (`image/jpeg`, `image/png`) | Yes | Registered reference photo of the resident |
| `capture_image` | File (`image/jpeg`, `image/png`) | Yes | Live camera capture at the hostel entrance |
| `threshold` | Float | No (Default: `0.55`) | Cosine similarity threshold for verification |

#### Successful Verification Response (`200 OK`):
```json
{
  "verified": true,
  "similarity_score": 0.9315,
  "threshold": 0.55,
  "face_detected": {
    "base_image": true,
    "capture_image": true
  },
  "message": "Face verified successfully"
}
```

#### Impostor / Mismatch Response (`200 OK`):
```json
{
  "verified": false,
  "similarity_score": -0.0382,
  "threshold": 0.55,
  "face_detected": {
    "base_image": true,
    "capture_image": true
  },
  "message": "Face verification failed: low similarity score"
}
```

#### No Face Detected Response (`200 OK`):
```json
{
  "verified": false,
  "similarity_score": 0.0,
  "threshold": 0.55,
  "face_detected": {
    "base_image": true,
    "capture_image": false
  },
  "message": "Captured image issue: No face detected in the image. Please ensure the face is clearly visible."
}
```

#### Multiple Faces Detected Response (`200 OK`):
```json
{
  "verified": false,
  "similarity_score": 0.0,
  "threshold": 0.55,
  "face_detected": {
    "base_image": true,
    "capture_image": true
  },
  "message": "Captured image issue: Multiple faces (2) detected in the image. Exactly one face must be present."
}
```

---

## Frontend Integration (JavaScript `fetch` Example)

The frontend web application or mobile app (React, React Native, Vue, Flutter, Vanilla JS) sends a single `FormData` request:

```javascript
async function verifyResident(baseImageFile, capturedBlob) {
  const formData = new FormData();
  formData.append('base_image', baseImageFile, 'base.jpg');
  formData.append('capture_image', capturedBlob, 'capture.jpg');
  formData.append('threshold', '0.55');

  try {
    const response = await fetch('http://127.0.0.1:8000/face/verify', {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();

    if (result.verified) {
      console.log('? Entry Approved! Similarity:', result.similarity_score);
      // Trigger hostel gate open / mark attendance
      showAccessGrantedUI(result.similarity_score);
    } else {
      console.warn('? Entry Denied:', result.message);
      // Display friendly error or ask student to look directly into camera
      showAccessDeniedUI(result.message);
    }
  } catch (error) {
    console.error('Network or server error:', error);
  }
}
```

---

## Metric Evaluation: FAR, FRR & Threshold Tuning

1. **False Accept Rate (FAR)**:
   $$\text{FAR} = \frac{\text{False Positives}}{\text{Total Impostor Attempts}}$$
   Proportion of unauthorized non-residents who are mistakenly verified. In a hostel setting, high FAR creates a security risk.

2. **False Reject Rate (FRR)**:
   $$\text{FRR} = \frac{\text{False Negatives}}{\text{Total Genuine Attempts}}$$
   Proportion of actual hostel residents who are incorrectly rejected. High FRR causes gate delays and inconvenience.

3. **Configuring the Threshold**:
   - Standard ArcFace models produce similarity $\approx 0.70 - 0.95$ for genuine matches and $< 0.30$ for different identities.
   - **Default Threshold ($0.55$)**: Well-balanced for day-to-day hostel gate operations.
   - **High Security Mode ($0.65$)**: Lower FAR, requires clearer lighting/alignment.
   - **Lenient Mode ($0.45$)**: Lower FRR, helpful if entrance camera has lower resolution.

---

## Security & Biometric Privacy Notes

- **Transient Processing**: Never store raw captured live frames permanently; process in memory and discard.
- **Embedding Storage**: When adding database persistence later, only store the mathematical 512-dimensional vector embedding, not the resident's raw biometric images.
- **HTTPS Enforcement**: Always deploy with TLS/HTTPS in production to prevent biometric intercept.
