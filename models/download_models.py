import os
import urllib.request
import zipfile
import io

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

ARCFACE_MODEL_PATH = os.path.join(MODELS_DIR, 'arcface_model.onnx')
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, 'face_detection_yunet.onnx')

def download_file(url, dest_path, desc):
    print(f'Downloading {desc} from {url}...')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024 * 64
        downloaded = 0
        with open(dest_path, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    print(f'\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)', end='')
        print(f'\nSaved to {dest_path}')

def setup_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Download YuNet Face Detector (232 KB)
    yunet_url = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
    if not os.path.exists(YUNET_MODEL_PATH) or os.path.getsize(YUNET_MODEL_PATH) < 200000:
        download_file(yunet_url, YUNET_MODEL_PATH, 'YuNet Face Detector')
    else:
        print(f'YuNet model already exists at {YUNET_MODEL_PATH}')

    # 2. Download and extract ArcFace model (w600k_mbf.onnx, ~13.6 MB)
    if not os.path.exists(ARCFACE_MODEL_PATH) or os.path.getsize(ARCFACE_MODEL_PATH) < 10000000:
        zip_url = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip'
        print('Downloading ArcFace weights archive (buffalo_s.zip)...')
        req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        print('Extracting w600k_mbf.onnx from archive...')
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with z.open('w600k_mbf.onnx') as src, open(ARCFACE_MODEL_PATH, 'wb') as dst:
                dst.write(src.read())
        print(f'ArcFace ONNX model saved to {ARCFACE_MODEL_PATH} ({os.path.getsize(ARCFACE_MODEL_PATH)} bytes)')
    else:
        print(f'ArcFace model already exists at {ARCFACE_MODEL_PATH}')

if __name__ == '__main__':
    setup_models()
