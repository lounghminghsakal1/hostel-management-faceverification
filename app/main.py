from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict
import uvicorn

from app.face_service import verify_face, DEFAULT_VERIFICATION_THRESHOLD
from app.face_model import get_face_model

# Initialize FastAPI application
app = FastAPI(
    title='Hostel Management - Face Verification API',
    description='Production-ready ArcFace Biometric Face Verification System for Hostel Access Control.',
    version='1.0.0'
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
    similarity_score: float = Field(description='Cosine similarity score between embeddings [-1.0 to 1.0]')
    face_matching_percentage: float = Field(description='Face match percentage [0 to 100], derived from similarity_score')
    threshold: float = Field(description='Cosine similarity threshold applied')
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
async def health_check():
    model = get_face_model()
    return HealthResponse(
        status='healthy',
        model_loaded=(model is not None),
        version='1.0.0'
    )

class VerifyRequest(BaseModel):
    base_image: str = Field(..., description="Public or accessible URL of the registered base photo")
    captured_image: Optional[str] = Field(default=None, description="Public or accessible URL of the live captured photo")
    capture_image: Optional[str] = Field(default=None, description="Alternative field name for captured photo URL")
    threshold: Optional[float] = Field(default=DEFAULT_VERIFICATION_THRESHOLD, description="Verification similarity threshold (default: 0.50)")

    def get_captured_url(self) -> str:
        url = self.captured_image or self.capture_image
        if not url:
            raise ValueError("Field 'captured_image' (or 'capture_image') URL is required.")
        return url

@app.post('/face/verify', response_model=VerifyResponse, tags=['Face Verification'])
async def api_verify_face(payload: VerifyRequest):
    """
    1:1 Face Verification Endpoint (URL-based).
    
    Receives JSON containing URLs for the registered Base Photo and Live Captured Photo:
    {
        "base_image": "https://example.com/student_base.jpg",
        "captured_image": "https://example.com/student_captured.jpg",
        "threshold": 0.50
    }
    Fetches both images over the internet, performs CNN-based face detection
    and ArcFace feature extraction, and returns whether both belong to the same person.
    """
    try:
        captured_url = payload.get_captured_url()
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )

    # Perform AI Face Verification using URLs directly
    result = verify_face(
        base_image=payload.base_image,
        capture_image=captured_url,
        threshold=payload.threshold if payload.threshold is not None else DEFAULT_VERIFICATION_THRESHOLD
    )

    return VerifyResponse(
        verified=result['verified'],
        similarity_score=result['similarity_score'],
        face_matching_percentage=result['face_matching_percentage'],
        threshold=result['threshold'],
        face_detected=FaceDetectionDetails(
            base_image=result['face_detected']['base_image'],
            capture_image=result['face_detected']['capture_image']
        ),
        message=result['message']
    )

if __name__ == '__main__':
    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=True)
