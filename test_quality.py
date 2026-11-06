"""
Checks the photo quality API (registration photo checks and selfie retakes).
Run: python test_quality.py
"""
import cv2
import numpy as np

from app.face_service import check_face_photo, verify_face

BASE = cv2.imread("test_images/base/student1_base.jpg")
SELFIE = cv2.imread("test_images/capture/student1_diff_lighting_angle.jpg")
TURNED = cv2.imread("test_images/base/student4_lena.jpg")  # head clearly turned
CROWD = cv2.imread("test_images/capture/multi_face_crowd.jpg")
BLANK = cv2.imread("test_images/capture/no_face_blank.jpg")


def codes(result):
    return [i["code"] for i in result["issues"]]


def check(name, condition, detail=""):
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}")
    assert condition, name


def main():
    good = check_face_photo(BASE)
    check("clear front-facing photo is accepted", good["ok"], f"({good['status']} {codes(good)})")

    cases = [
        ("blurry photo", cv2.GaussianBlur(BASE, (21, 21), 0), "BLURRY"),
        ("dark photo", (BASE * 0.25).astype(np.uint8), "TOO_DARK"),
        ("overexposed photo", np.clip(BASE.astype(np.float32) * 1.9 + 60, 0, 255).astype(np.uint8), "TOO_BRIGHT"),
        ("tiny photo", cv2.resize(BASE, (180, 225)), "FACE_TOO_SMALL"),
        ("sideways photo", cv2.rotate(BASE, cv2.ROTATE_90_CLOCKWISE), "PHOTO_SIDEWAYS"),
        ("head turned away", TURNED, "NOT_FACING_CAMERA"),
        ("no face", BLANK, "NO_FACE"),
        ("group photo", CROWD, "MULTIPLE_FACES"),
    ]
    for name, image, expected in cases:
        result = check_face_photo(image)
        check(f"{name} -> {expected}", not result["ok"] and expected in codes(result), f"({codes(result)})")
        check(f"{name} has a message", bool(result["message"]))

    unreadable = check_face_photo(b"not an image")
    check("unreadable file -> IMAGE_UNREADABLE", codes(unreadable) == ["IMAGE_UNREADABLE"])

    # Selfies are checked leniently: a soft or dim selfie is still compared...
    soft = verify_face(BASE, cv2.GaussianBlur(SELFIE, (15, 15), 0))
    check("slightly blurry selfie is still compared", soft["status"] == "VERIFIED", f"({soft['status']}, {soft['face_matching_percentage']}%)")
    dim = verify_face(BASE, (SELFIE * 0.35).astype(np.uint8))
    check("dim selfie is still compared", dim["status"] == "VERIFIED", f"({dim['status']}, {dim['face_matching_percentage']}%)")
    # ...but an unusable one is sent back for a retake with the reason
    far = np.full((2400, 1800, 3), 190, np.uint8)
    far[200:320, 1500:1590] = cv2.resize(SELFIE, (90, 120))
    far_result = verify_face(BASE, far)
    check("far-away selfie -> retake", far_result["status"] in ("LOW_QUALITY_CAPTURE", "NO_FACE_IN_CAPTURE"),
          f"({far_result['status']}: {far_result['message']})")
    black = verify_face(BASE, (SELFIE * 0.05).astype(np.uint8))
    check("pitch-black selfie -> retake", black["status"] in ("LOW_QUALITY_CAPTURE", "NO_FACE_IN_CAPTURE"),
          f"({black['status']}: {black['message']})")
    sideways = verify_face(BASE, cv2.rotate(SELFIE, cv2.ROTATE_90_COUNTERCLOCKWISE))
    check("sideways selfie is turned upright and matched", sideways["status"] == "VERIFIED", f"({sideways['status']})")

    print("\nAll quality checks passed.")


if __name__ == "__main__":
    main()
