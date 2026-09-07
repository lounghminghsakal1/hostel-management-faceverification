import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional, Dict, Any, List

# Standard reference 5-point facial coordinates for 112x112 ArcFace input
REFERENCE_POINTS_112 = np.array([
    [38.2946, 51.6963],  # Subject's right eye (Observer left)
    [73.5318, 51.5014],  # Subject's left eye (Observer right)
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Subject's right mouth corner (Observer left)
    [70.7299, 92.2041]   # Subject's left mouth corner (Observer right)
], dtype=np.float32)

class FaceModel:
    """
    Handles Face Detection using OpenCV YuNet and
    Face Feature Embedding generation using Pretrained ArcFace ONNX.
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

        # Initialize ArcFace ONNX Session
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        self.arcface_session = ort.InferenceSession(
            self.arcface_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.arcface_session.get_inputs()[0].name
        self.output_name = self.arcface_session.get_outputs()[0].name

        # Initialize YuNet Face Detector
        # Default placeholder size (320, 320), updated per input image
        self.detector = cv2.FaceDetectorYN.create(
            model=self.yunet_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )
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
                        raise ValueError(f"HTTP {resp.status_code} while downloading image from: {image_input}")
                    nparr = np.frombuffer(resp.content, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError(f"Failed to decode image downloaded from: {image_input}")
                    return img
                except requests.RequestException as req_err:
                    raise ValueError(f"Network error fetching image from {image_input}: {str(req_err)}")
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
        Detects faces in the given BGR image using YuNet.
        Returns dict containing:
          - count: number of faces found
          - status: 'SUCCESS', 'NO_FACE_DETECTED', or 'MULTIPLE_FACES_DETECTED'
          - bbox: [x, y, w, h] of best face (if found)
          - landmarks: 5-point landmarks (if found)
          - confidence: detection confidence score
        """
        h, w, _ = image_bgr.shape
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(image_bgr)

        if faces is None or len(faces) == 0:
            return {
                "count": 0,
                "status": "NO_FACE_DETECTED",
                "message": "No face detected in the image. Please ensure the face is clearly visible."
            }

        count = len(faces)
        if count > 1:
            return {
                "count": count,
                "status": "MULTIPLE_FACES_DETECTED",
                "message": f"Multiple faces ({count}) detected in the image. Exactly one face must be present."
            }

        face_data = faces[0]
        # Bounding box: [x, y, w, h]
        bbox = [int(val) for val in face_data[0:4]]
        confidence = float(face_data[14])

        # 5 facial landmarks:
        # [x_re, y_re], [x_le, y_le], [x_nt, y_nt], [x_rc, y_rc], [x_lc, y_lc]
        landmarks = np.array([
            [face_data[4], face_data[5]],   # Right eye (subject)
            [face_data[6], face_data[7]],   # Left eye (subject)
            [face_data[8], face_data[9]],   # Nose tip
            [face_data[10], face_data[11]], # Right mouth corner
            [face_data[12], face_data[13]]  # Left mouth corner
        ], dtype=np.float32)

        return {
            "count": 1,
            "status": "SUCCESS",
            "bbox": bbox,
            "landmarks": landmarks,
            "confidence": confidence,
            "message": "Single face detected successfully."
        }

    def align_and_crop_face(self, image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Aligns and crops the face to 112x112 standard input using similarity transform
        computed from the 5 facial landmarks.
        """
        # Estimate partial affine similarity matrix (rotation, translation, uniform scale)
        M, _ = cv2.estimateAffinePartial2D(landmarks, REFERENCE_POINTS_112, method=cv2.RANSAC)

        if M is None:
            # Fallback if affine estimation fails: simple centered crop/resize
            return cv2.resize(image_bgr, (112, 112), interpolation=cv2.INTER_AREA)

        # Warp face into canonical 112x112 space
        aligned_face = cv2.warpAffine(
            image_bgr,
            M,
            (112, 112),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        return aligned_face

    def generate_embedding(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """
        Computes the 512-dimensional L2-normalized feature embedding from an aligned 112x112 face.
        """
        # Preprocessing: convert to float32, normalize to [-1, 1]
        # InsightFace MobileFaceNet / ArcFace expects BGR with (x - 127.5) / 127.5
        face_blob = aligned_face_bgr.astype(np.float32)
        face_blob = (face_blob - 127.5) / 127.5

        # Transpose from HWC (112, 112, 3) to CHW (3, 112, 112)
        face_blob = np.transpose(face_blob, (2, 0, 1))

        # Add batch dimension: (1, 3, 112, 112)
        face_blob = np.expand_dims(face_blob, axis=0)

        # Run ONNX inference
        outputs = self.arcface_session.run([self.output_name], {self.input_name: face_blob})
        embedding = outputs[0][0]  # shape: (512,)

        # L2-normalization (unit hypersphere projection)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

# Global singleton instance for efficient reuse across API calls
_global_face_model: Optional[FaceModel] = None

def get_face_model() -> FaceModel:
    global _global_face_model
    if _global_face_model is None:
        _global_face_model = FaceModel()
    return _global_face_model
