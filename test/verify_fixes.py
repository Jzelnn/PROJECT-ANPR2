import sys
import os
import base64
import numpy as np
import cv2

# Set path for backend imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
import config

def test_annotated_image():
    print("Testing generate_annotated_frame...")
    from app import generate_annotated_frame
    
    # Create a synthetic image 640x480
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_img[:] = (50, 50, 50)
    
    detections = [{
        "track_id": 1,
        "vehicle_type": "car",
        "body_style": "SUV",
        "vehicle_confidence": 0.95,
        "bbox": [100, 100, 400, 400],
        "plate_bbox": [200, 300, 320, 340],
        "license_plate": "B 1234 ABC",
        "ocr_confidence": 0.92,
        "inside_interest_area": True,
        "lost_interest": False
    }]
    
    data_uri = generate_annotated_frame(dummy_img, detections, frame_id=101)
    assert data_uri is not None, "data_uri is None"
    assert data_uri.startswith("data:image/jpeg;base64,"), f"Invalid prefix: {data_uri[:30]}"
    
    b64_data = data_uri.split(",", 1)[1]
    raw_bytes = base64.b64decode(b64_data)
    assert raw_bytes[:2] == b'\xff\xd8', "JPEG header magic bytes mismatch"
    
    decoded_img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded_img is not None, "Failed to decode generated JPEG"
    assert decoded_img.shape == (480, 640, 3), f"Shape mismatch: {decoded_img.shape}"
    print("[PASS] generate_annotated_frame verified successfully!")

def test_plate_refinement():
    print("Testing refine_indonesian_plate...")
    from app import refine_indonesian_plate
    
    # Check that '225 BK' is not converted to 'Z 225 BK'
    res = refine_indonesian_plate("225 BK")
    print(f"refine_indonesian_plate('225 BK') -> {res}")
    assert "Z 225 BK" not in res, f"Expected no 'Z' prefix hallucination, got {res}"
    
    # Check standard plate
    res_b = refine_indonesian_plate("B 1958 RZH")
    print(f"refine_indonesian_plate('B 1958 RZH') -> {res_b}")
    assert res_b == "B 1958 RZH", f"Expected B 1958 RZH, got {res_b}"
    print("[PASS] refine_indonesian_plate verified successfully!")

def test_track_isolation():
    print("Testing track lifecycle & isolation...")
    from app import VehicleConfirmationManager
    
    cm = VehicleConfirmationManager()
    
    # Vehicle A appears
    tid_a, trk_a = cm.get_or_create_track(1, [100, 100, 300, 300], 1920, 1080, frame_id=1)
    trk_a.cached_vtype = "SUV"
    trk_a.last_plate_bbox = [150, 250, 250, 280]
    trk_a.best_candidate_display = "B 1234 ABC"
    trk_a.best_candidate_conf = 0.95
    trk_a.is_locked = True
    
    # Vehicle A leaves / loses interest
    trk_a.lost_interest = True
    trk_a.lifecycle_state = "LOST"
    
    # Vehicle B enters with track_id 2
    tid_b, trk_b = cm.get_or_create_track(2, [120, 120, 320, 320], 1920, 1080, frame_id=50)
    assert tid_b == 2
    assert trk_b.cached_vtype is None, f"Vehicle B inherited cached_vtype: {trk_b.cached_vtype}"
    assert trk_b.last_plate_bbox is None, f"Vehicle B inherited plate_bbox: {trk_b.last_plate_bbox}"
    assert trk_b.best_candidate is None, f"Vehicle B inherited best_candidate: {trk_b.best_candidate}"
    assert trk_b.best_candidate_display is None, f"Vehicle B inherited display plate: {trk_b.best_candidate_display}"
    assert trk_b.is_locked is False, "Vehicle B inherited locked state"
    print("[PASS] Track isolation verified successfully!")

if __name__ == "__main__":
    test_annotated_image()
    test_plate_refinement()
    test_track_isolation()
    print("ALL TESTS PASSED!")
