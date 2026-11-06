from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

from app.face_service import verify_face, DEFAULT_VERIFICATION_THRESHOLD, MAX_REFERENCE_IMAGES
from app.face_model import get_face_model

# Initialize FastAPI application
app = FastAPI(
    title='Hostel Management - Face Verification API',
    description='Production-ready ArcFace Biometric Face Verification System for Hostel Access Control.',
    version='1.1.0'
)

# Configure CORS for seamless frontend web & mobile integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Configure specific origins for production deployment
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Pydantic Response Schemas
class FaceDetectionDetails(BaseModel):
    base_image: bool = Field(description='Whether a single face was detected in base image')
    capture_image: bool = Field(description='Whether a single face was detected in captured image')

class VerifyResponse(BaseModel):
    verified: bool = Field(description='True if both images belong to the same person, False otherwise')
    status: str = Field(description='VERIFIED, NOT_VERIFIED, or why it could not be checked (e.g. NO_FACE_IN_CAPTURE)')
    similarity_score: float = Field(description='Final cosine similarity [-1.0 to 1.0] (registered photo, softened by recent check-ins)')
    base_similarity: float = Field(description='Cosine similarity against the registered photo only')
    reference_similarity: Optional[float] = Field(default=None, description='Best cosine similarity against recent check-in selfies, if any were sent')
    matched_with: Optional[str] = Field(default=None, description='"base_photo" or "recent_check_ins"')
    references_used: int = Field(default=0, description='How many reference selfies had a usable face')
    face_matching_percentage: float = Field(description='Calibrated match percentage [0 to 100]: 65+ trusted, 30-65 check, below 30 likely someone else')
    threshold: float = Field(description='Cosine similarity threshold applied for `verified`')
    face_detected: FaceDetectionDetails
    message: str = Field(description='User-friendly result or diagnostic message')

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

@app.on_event('startup')
async def startup_event():
    # Warm up models on server launch
    try:
        get_face_model()
        print('FastAPI Startup: FaceModel preloaded successfully.')
    except Exception as e:
        print(f'FastAPI Startup Error loading models: {e}')

@app.get('/', tags=['Health Check'])
async def root():
    return {
        'service': 'Hostel Management Face Verification API',
        'status': 'online',
        'docs_url': '/docs'
    }

@app.get('/health', response_model=HealthResponse, tags=['Health Check'])
def health_check():
    model = get_face_model()
    return HealthResponse(
        status='healthy',
        model_loaded=(model is not None),
        version='1.1.0'
    )

class VerifyRequest(BaseModel):
    base_image: str = Field(..., description="Public or accessible URL of the registered base photo")
    captured_image: str = Field(..., description="Public or accessible URL of the live captured photo")
    reference_images: List[str] = Field(
        default_factory=list,
        max_length=10,
        description=f"Optional URLs of recent check-in selfies that matched the base photo well (up to {MAX_REFERENCE_IMAGES} are used)",
    )
    threshold: Optional[float] = Field(default=DEFAULT_VERIFICATION_THRESHOLD, description="Verification cosine threshold (default: 0.32, the 65% line)")

# A plain `def` endpoint: FastAPI runs it in a worker thread, so one slow
# download or inference doesn't block every other student checking in.
@app.post('/face/verify', response_model=VerifyResponse, tags=['Face Verification'])
def api_verify_face(payload: VerifyRequest):
    """
    1:1 Face Verification Endpoint (URL-based).

    Receives JSON containing URLs for the registered Base Photo and Live Captured Photo:
    {
        "base_image": "https://example.com/student_base.jpg",
        "captured_image": "https://example.com/student_captured.jpg",
        "reference_images": ["https://example.com/recent_checkin_1.jpg"]
    }
    Fetches the images, performs CNN-based face detection and ArcFace feature
    extraction, and returns whether both belong to the same person.
    """
    result = verify_face(
        base_image=payload.base_image,
        capture_image=payload.captured_image,
        threshold=payload.threshold if payload.threshold is not None else DEFAULT_VERIFICATION_THRESHOLD,
        reference_images=payload.reference_images,
    )
    return VerifyResponse(**result)

if __name__ == '__main__':
    # Bind to localhost: only the Node backend on the same machine should call this
    uvicorn.run('app.main:app', host='127.0.0.1', port=8000, reload=True)
