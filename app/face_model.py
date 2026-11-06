import os
import threading
import cv2
import numpy as np
import onnxruntime as ort
from typing import Optional, Dict, Any

# Standard reference 5-point facial coordinates for 112x112 ArcFace input
REFERENCE_POINTS_112 = np.array([
    [38.2946, 51.6963],  # Subject's right eye (Observer left)
    [73.5318, 51.5014],  # Subject's left eye (Observer right)
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Subject's right mouth corner (Observer left)
    [70.7299, 92.2041]   # Subject's left mouth corner (Observer right)
], dtype=np.float32)

# Faces are detected on a copy no larger than this (long side, px). Phone photos
# can be 3000-4000 px; detection there is slow and not more accurate. Alignment
# still uses the full-resolution image.
DETECTION_MAX_SIDE = 1024

# A second face at least this big (relative to the largest face's area) means
# another person is clearly in the frame. Smaller faces (people far behind,
# posters) are ignored and the largest face is used.
SECOND_FACE_AREA_RATIO = 0.5


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Least-squares similarity transform (rotation, uniform scale, translation)
    mapping src points onto dst points (Umeyama, 1991). Uses all 5 landmarks,
    like InsightFace's own alignment, instead of RANSAC which can drop one.
    Returns a 2x3 matrix for cv2.warpAffine.
    """
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    n = src.shape[0]
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / n
    u, sigma, vt = np.linalg.svd(cov)
    d = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1, 1] = -1
    rotation = u @ d @ vt
    src_var = (src_c ** 2).sum() / n
    scale = (sigma * np.diag(d)).sum() / src_var if src_var > 0 else 1.0
    translation = dst_mean - scale * rotation @ src_mean
    return np.hstack([scale * rotation, translation[:, None]]).astype(np.float32)


class FaceModel:
    """
    Handles Face Detection using OpenCV YuNet and
    Face Feature Embedding generation using Pretrained ArcFace ONNX
    (InsightFace w600k_mbf, from the buffalo_s pack).
    """

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

        self.models_dir = models_dir
        self.arcface_path = os.path.join(models_dir, "arcface_model.onnx")
        self.yunet_path = os.path.join(models_dir, "face_detection_yunet.onnx")

        # Verify model files exist
        if not os.path.exists(self.arcface_path):
            raise FileNotFoundError(f"ArcFace model not found at {self.arcface_path}. Run models/download_models.py first.")
        if not os.path.exists(self.yunet_path):
            raise FileNotFoundError(f"YuNet model not found at {self.yunet_path}. Run models/download_models.py first.")

        # Initialize ArcFace ONNX Session (InferenceSession.run is thread-safe)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        self.arcface_session = ort.InferenceSession(
            self.arcface_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.arcface_session.get_inputs()[0].name
        self.output_name = self.arcface_session.get_outputs()[0].name

        # Initialize YuNet Face Detector. It keeps per-call state (input size),
        # so concurrent requests take turns through a lock.
        self.detector = cv2.FaceDetectorYN.create(
            model=self.yunet_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )
        self._detector_lock = threading.Lock()
        print("FaceModel initialized successfully: ArcFace ONNX & YuNet Detector loaded.")

    def decode_image(self, image_input: Any) -> np.ndarray:
        """
        Decodes an image from URL (http/https), filepath (str), bytes, or directly returns numpy array.
        Returns BGR numpy image.
        """
        if isinstance(image_input, np.ndarray):
            return image_input
        elif isinstance(image_input, (bytes, bytearray)):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from bytes. Ensure valid image file (JPEG, PNG).")
            return img
        elif isinstance(image_input, str):
            # Check if input is a URL
            if image_input.startswith(("http://", "https://")):
                import requests
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                try:
                    resp = requests.get(image_input, headers=headers, timeout=15)
                    if resp.status_code != 200:
                        # Don't echo the URL: it's a signed S3 link
                        raise ValueError(f"HTTP {resp.status_code} while downloading the image")
                    nparr = np.frombuffer(resp.content, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("Downloaded file is not a valid image")
                    return img
                except requests.RequestException as req_err:
                    raise ValueError(f"Network error fetching image: {type(req_err).__name__}")
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            img = cv2.imread(image_input, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to read image at path: {image_input}")
            return img
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def detect_face(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Detects faces in the given BGR image using YuNet and picks the largest one.
        Returns dict containing:
          - count: number of faces found
          - status: 'SUCCESS', 'NO_FACE_DETECTED', or 'MULTIPLE_FACES_DETECTED'
            (MULTIPLE only when a second face is nearly as big as the largest)
          - bbox: [x, y, w, h] of the chosen face, in original image pixels
          - landmarks: 5-point landmarks of the chosen face, in original image pixels
          - confidence: detection confidence score
        """
        h, w = image_bgr.shape[:2]
        scale = min(1.0, DETECTION_MAX_SIDE / float(max(h, w)))
        small = image_bgr if scale == 1.0 else cv2.resize(
            image_bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
        )
        sh, sw = small.shape[:2]

        with self._detector_lock:
            self.detector.setInputSize((sw, sh))
            _, faces = self.detector.detect(small)

        if faces is None or len(faces) == 0:
            return {
                "count": 0,
                "status": "NO_FACE_DETECTED",
                "message": "No face detected in the image. Please ensure the face is clearly visible."
            }

        faces = sorted(faces, key=lambda f: float(f[2] * f[3]), reverse=True)
        count = len(faces)
        largest_area = float(faces[0][2] * faces[0][3])
        if count > 1 and float(faces[1][2] * faces[1][3]) >= SECOND_FACE_AREA_RATIO * largest_area:
            return {
                "count": count,
                "status": "MULTIPLE_FACES_DETECTED",
                "message": f"Multiple faces ({count}) detected in the image. Exactly one face must be present."
            }

        face_data = faces[0]
        inv = 1.0 / scale
        bbox = [int(round(float(val) * inv)) for val in face_data[0:4]]
        confidence = float(face_data[14])

        # 5 facial landmarks:
        # [x_re, y_re], [x_le, y_le], [x_nt, y_nt], [x_rc, y_rc], [x_lc, y_lc]
        landmarks = np.array(face_data[4:14], dtype=np.float32).reshape(5, 2) * inv

        return {
            "count": count,
            "status": "SUCCESS",
            "bbox": bbox,
            "landmarks": landmarks,
            "confidence": confidence,
            "message": "Single face detected successfully."
        }

    def align_and_crop_face(self, image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Aligns and crops the face to the 112x112 ArcFace input using a
        least-squares similarity transform over all 5 facial landmarks.
        """
        M = similarity_transform(landmarks, REFERENCE_POINTS_112)
        return cv2.warpAffine(
            image_bgr,
            M,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

    def _embed_once(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        # InsightFace models are trained on RGB input normalised as (x - 127.5) / 127.5
        rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis]  # (1, 3, 112, 112)
        return self.arcface_session.run([self.output_name], {self.input_name: blob})[0][0]

    def generate_embedding(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """
        Computes the 512-dimensional L2-normalized feature embedding from an aligned 112x112 face.
        The embeddings of the face and its mirror image are averaged ("flip test"),
        which makes the result steadier for selfies (front cameras may mirror)
        and small head turns.
        """
        embedding = self._embed_once(aligned_face_bgr) + self._embed_once(cv2.flip(aligned_face_bgr, 1))

        # L2-normalization (unit hypersphere projection)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

# Global singleton instance for efficient reuse across API calls
_global_face_model: Optional[FaceModel] = None
_global_lock = threading.Lock()

def get_face_model() -> FaceModel:
    global _global_face_model
    if _global_face_model is None:
        with _global_lock:
            if _global_face_model is None:
                _global_face_model = FaceModel()
    return _global_face_model
