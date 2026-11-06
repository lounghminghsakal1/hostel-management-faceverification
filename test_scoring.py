"""
Checks the calibrated percentage and the recent-selfie references.
Run: python test_scoring.py
"""
from app.face_service import (
    REFERENCE_BOOST_CAP,
    compute_face_matching_percentage,
    verify_face,
)

BASE = "test_images/base/student1_base.jpg"
SAME = "test_images/capture/student1_diff_lighting_angle.jpg"
OTHER = "test_images/capture/student2_impostor.jpg"
CROWD = "test_images/capture/multi_face_crowd.jpg"
BLANK = "test_images/capture/no_face_blank.jpg"


def check(name, condition, detail=""):
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}")
    assert condition, name


def main():
    # Calibration bands: red < 30 <= yellow < 65 <= green
    check("cosine 0.00 -> 0%", compute_face_matching_percentage(0.0) == 0.0)
    check("cosine 0.20 -> 30%", compute_face_matching_percentage(0.20) == 30.0)
    check("cosine 0.32 -> 65%", compute_face_matching_percentage(0.32) == 65.0)
    check("cosine 0.90 -> 100%", compute_face_matching_percentage(0.90) == 100.0)

    same = verify_face(BASE, SAME)
    check("same person is trusted (>= 65%)", same["face_matching_percentage"] >= 65,
          f"({same['face_matching_percentage']}%, cosine {same['base_similarity']})")

    other = verify_face(BASE, OTHER)
    check("different person is red (< 30%)", other["face_matching_percentage"] < 30,
          f"({other['face_matching_percentage']}%, cosine {other['base_similarity']})")

    # A recent selfie can lift the score, but only by REFERENCE_BOOST_CAP above the base score
    boosted = verify_face(BASE, SAME, reference_images=[SAME])
    expected = min(1.0, round(same["base_similarity"] + REFERENCE_BOOST_CAP, 4))
    check("reference boost is capped", abs(boosted["similarity_score"] - expected) < 1e-3,
          f"(base {boosted['base_similarity']}, final {boosted['similarity_score']}, via {boosted['matched_with']})")

    # The student's own recent selfies don't help an impostor
    impostor = verify_face(BASE, OTHER, reference_images=[SAME])
    check("impostor stays red with genuine references", impostor["face_matching_percentage"] < 30,
          f"({impostor['face_matching_percentage']}%)")

    # Even a "poisoned" reference (the impostor's own face) can't lift them past red
    poisoned = verify_face(BASE, OTHER, reference_images=[OTHER])
    check("impostor stays red with their own face as reference", poisoned["face_matching_percentage"] < 30,
          f"({poisoned['face_matching_percentage']}%)")

    check("blank selfie -> NO_FACE_IN_CAPTURE", verify_face(BASE, BLANK)["status"] == "NO_FACE_IN_CAPTURE")
    crowd = verify_face(BASE, CROWD)["status"]
    check("crowd selfie -> MULTIPLE_FACES_IN_CAPTURE or a low score", crowd in ("MULTIPLE_FACES_IN_CAPTURE", "NOT_VERIFIED"), f"({crowd})")

    print("\nAll scoring checks passed.")


if __name__ == "__main__":
    main()
