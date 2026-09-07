import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_fastapi_endpoints():
    print('======================================================================')
    print('          FASTAPI /face/verify URL-BASED ENDPOINT TEST                ')
    print('======================================================================\n')

    # 1. Health check
    res_health = client.get('/health')
    print(f'GET /health -> Status: {res_health.status_code}, Body: {res_health.json()}')
    assert res_health.status_code == 200

    # 2. Same Person verification via POST /face/verify (using URLs)
    url_base = 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg'
    url_diff = 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama2.jpg'
    
    payload_same = {
        'base_image': url_base,
        'captured_image': url_diff,
        'threshold': 0.50
    }
    resp = client.post('/face/verify', json=payload_same)
    print('\nPOST /face/verify [Same Person URL Test]:')
    print(f'HTTP Status: {resp.status_code}')
    print('Response JSON:', json.dumps(resp.json(), indent=2))
    assert resp.status_code == 200
    assert resp.json()['verified'] is True
    assert resp.json()['similarity_score'] >= 0.50

    # 3. Impostor verification via POST /face/verify (using URLs)
    url_impostor = 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg'
    payload_diff = {
        'base_image': url_base,
        'captured_image': url_impostor,
        'threshold': 0.50
    }
    resp = client.post('/face/verify', json=payload_diff)
    print('\nPOST /face/verify [Impostor URL Test]:')
    print(f'HTTP Status: {resp.status_code}')
    print('Response JSON:', json.dumps(resp.json(), indent=2))
    assert resp.status_code == 200
    assert resp.json()['verified'] is False

    print('\n>>> ALL FASTAPI URL-BASED TESTS PASSED SUCCESSFULLY! <<<\n')

if __name__ == '__main__':
    test_fastapi_endpoints()
