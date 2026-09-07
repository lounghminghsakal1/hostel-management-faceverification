import os
import json
from app.face_service import verify_face, compute_cosine_similarity, compute_euclidean_distance
from app.face_model import get_face_model

def run_tests():
    print("======================================================================")
    print("   HOSTEL MANAGEMENT - AI FACE VERIFICATION TEST SUITE (PHASE 8)     ")
    print("======================================================================\n")

    test_cases = [
        {
            "name": "1. Same Person - Identical Image (Obama vs Obama)",
            "base": "test_images/base/student1_base.jpg",
            "capture": "test_images/base/student1_base.jpg",
            "expected_verified": True,
            "expected_similarity_min": 0.98
        },
        {
            "name": "2. Same Person - Different Photo / Lighting / Angle",
            "base": "test_images/base/student1_base.jpg",
            "capture": "test_images/capture/student1_diff_lighting_angle.jpg",
            "expected_verified": True,
            "expected_similarity_min": 0.55
        },
        {
            "name": "3. Same Person - Beard & Glasses Variation",
            "base": "test_images/base/student3_beard_glasses_base.png",
            "capture": "test_images/capture/student3_beard_glasses_capture.jpg",
            "expected_verified": True,
            "expected_similarity_min": 0.75
        },
        {
            "name": "4. Different Persons - Impostor Test (Obama vs Biden)",
            "base": "test_images/base/student1_base.jpg",
            "capture": "test_images/capture/student2_impostor.jpg",
            "expected_verified": False,
            "expected_similarity_max": 0.40
        },
        {
            "name": "5. Edge Case - No Face in Capture Image",
            "base": "test_images/base/student1_base.jpg",
            "capture": "test_images/capture/no_face_blank.jpg",
            "expected_verified": False
        },
        {
            "name": "6. Edge Case - Multiple Faces in Capture Frame",
            "base": "test_images/base/student1_base.jpg",
            "capture": "test_images/capture/multi_face_crowd.jpg",
            "expected_verified": False
        }
    ]

    all_passed = True
    header_scenario = "Scenario"
    header_verified = "Verified"
    header_sim = "Similarity"
    header_result = "Result"
    print(f"| {header_scenario:<45} | {header_verified:<9} | {header_sim:<11} | {header_result:<8} |")
    print("|" + "-"*47 + "|" + "-"*11 + "|" + "-"*13 + "|" + "-"*10 + "|")

    for tc in test_cases:
        res = verify_face(tc["base"], tc["capture"], threshold=0.55)
        passed = (res["verified"] == tc["expected_verified"])
        if "expected_similarity_min" in tc:
            passed = passed and (res["similarity_score"] >= tc["expected_similarity_min"])
        if "expected_similarity_max" in tc:
            passed = passed and (res["similarity_score"] <= tc["expected_similarity_max"])
        
        if not passed:
            all_passed = False

        status = "PASS [OK]" if passed else "FAIL [X]"
        name_str = tc["name"]
        ver_str = str(res["verified"])
        score_val = res["similarity_score"]
        print(f"| {name_str:<45} | {ver_str:<9} | {score_val:>11.4f} | {status:<8} |")
        if not res["verified"]:
            print(f"   +-- Reason/Message: {res['message']}")

    print("\n======================================================================")
    print("                      EVALUATION METHODOLOGY                          ")
    print("======================================================================\n")
    print("In Face Verification, we evaluate two fundamental operational metrics:\n")
    print("1. False Accept Rate (FAR):")
    print("   - Percentage of unauthorized/different people incorrectly marked as 'Verified'.")
    print("   - FAR = (False Positives) / (Total Impostor Attempts)")
    print("   - In a hostel, high FAR is a SECURITY RISK (unauthorized entry allowed).\n")
    print("2. False Reject Rate (FRR):")
    print("   - Percentage of legitimate residents incorrectly marked as 'Not Verified'.")
    print("   - FRR = (False Negatives) / (Total Resident Attempts)")
    print("   - High FRR causes student INCONVENIENCE (frequent retries at gate).\n")
    print("3. Optimal Threshold Selection:")
    print("   - Increasing the threshold (e.g. from 0.55 to 0.65) lowers FAR (tighter security) but may increase FRR.")
    print("   - Lowering the threshold (e.g. to 0.45) lowers FRR (fewer retries) but increases FAR.")
    print("   - Recommended procedure: Collect 50-100 pairs of genuine resident entries and 50-100 pairs of impostor cross-matches.")
    print("     Plot the ROC curve to choose threshold at the Equal Error Rate (EER) or targeting FAR < 0.1%.\n")

    if all_passed:
        print(">>> ALL 6 BENCHMARK SCENARIOS PASSED SUCCESSFULLY! <<<\n")
    else:
        print(">>> SOME SCENARIOS FAILED. <<<\n")

if __name__ == "__main__":
    run_tests()
