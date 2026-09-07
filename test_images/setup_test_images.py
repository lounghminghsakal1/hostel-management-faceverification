import os
import requests
import cv2
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), 'base')
CAPTURE_DIR = os.path.join(os.path.dirname(__file__), 'capture')

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)

SAMPLES = {
    os.path.join(BASE_DIR, 'student1_base.jpg'): 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg',
    os.path.join(CAPTURE_DIR, 'student1_diff_lighting_angle.jpg'): 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama2.jpg',
    os.path.join(CAPTURE_DIR, 'student2_impostor.jpg'): 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg',
    os.path.join(BASE_DIR, 'student3_beard_glasses_base.png'): 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/alex-lacamoire.png',
    os.path.join(CAPTURE_DIR, 'multi_face_crowd.jpg'): 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/two_people.jpg',
    os.path.join(BASE_DIR, 'student4_lena.jpg'): 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
}

for path, url in SAMPLES.items():
    if not os.path.exists(path):
        print(f'Fetching {os.path.basename(path)}...')
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            with open(path, 'wb') as f:
                f.write(resp.content)
            print(f'Saved {path}')
        else:
            print(f'Failed to fetch {url}: {resp.status_code}')

# Generate a synthetic "no face" blank scenery image
no_face_path = os.path.join(CAPTURE_DIR, 'no_face_blank.jpg')
if not os.path.exists(no_face_path):
    blank_img = np.zeros((400, 400, 3), dtype=np.uint8)
    blank_img[:] = (200, 180, 150) # Beige wall texture
    cv2.putText(blank_img, 'EMPTY HOSTEL ROOM', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    cv2.imwrite(no_face_path, blank_img)
    print(f'Generated {no_face_path}')

# Generate glasses / variation from student3 for captured variation
student3_base = os.path.join(BASE_DIR, 'student3_beard_glasses_base.png')
student3_capture = os.path.join(CAPTURE_DIR, 'student3_beard_glasses_capture.jpg')
if os.path.exists(student3_base) and not os.path.exists(student3_capture):
    img = cv2.imread(student3_base)
    # Apply minor brightness and color shift to simulate a real hostel gate camera
    adjusted = cv2.convertScaleAbs(img, alpha=0.92, beta=15)
    cv2.imwrite(student3_capture, adjusted)
    print(f'Generated {student3_capture}')

print('Test images setup complete!')
