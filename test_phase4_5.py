import numpy as np
from app.face_model import get_face_model

model = get_face_model()

print('\n--- PHASE 4: FACE DETECTION TESTS ---')
# 1. Single face detection
img1 = model.decode_image('test_images/base/student1_base.jpg')
res1 = model.detect_face(img1)
print(f"Test 1 (Single face): status = {res1['status']}, confidence = {res1['confidence']:.3f}, bbox = {res1['bbox']}")
assert res1['status'] == 'SUCCESS'

# 2. No face detection
img_blank = model.decode_image('test_images/capture/no_face_blank.jpg')
res_blank = model.detect_face(img_blank)
print(f"Test 2 (No face): status = {res_blank['status']}, msg = {res_blank['message']}")
assert res_blank['status'] == 'NO_FACE_DETECTED'

# 3. Multi face detection
img_multi = model.decode_image('test_images/capture/multi_face_crowd.jpg')
res_multi = model.detect_face(img_multi)
print(f"Test 3 (Multi face): status = {res_multi['status']}, count = {res_multi['count']}")
assert res_multi['status'] == 'MULTIPLE_FACES_DETECTED'

print('\n--- PHASE 5: FACE EMBEDDING TESTS ---')
# Generate Embedding 1
aligned1 = model.align_and_crop_face(img1, res1['landmarks'])
emb1 = model.generate_embedding(aligned1)
print(f"Embedding A: shape = {emb1.shape}, L2-norm = {np.linalg.norm(emb1):.4f}")
assert emb1.shape == (512,)
assert abs(np.linalg.norm(emb1) - 1.0) < 1e-4

# Generate Embedding 2
img2 = model.decode_image('test_images/capture/student1_diff_lighting_angle.jpg')
res2 = model.detect_face(img2)
aligned2 = model.align_and_crop_face(img2, res2['landmarks'])
emb2 = model.generate_embedding(aligned2)
print(f"Embedding B: shape = {emb2.shape}, L2-norm = {np.linalg.norm(emb2):.4f}")
assert emb2.shape == (512,)
assert abs(np.linalg.norm(emb2) - 1.0) < 1e-4

print('\nALL PHASE 4 & 5 ASSERTIONS PASSED SUCCESSFULLY!')
