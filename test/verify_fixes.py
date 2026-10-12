import sys
import os
import base64
import json
import threading
import time
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
    
    data_uri, raw_b64, mime_type = generate_annotated_frame(dummy_img, detections, frame_id=101)
    assert data_uri is not None, "data_uri is None"
    assert raw_b64 is not None, "raw_b64 is None"
    assert mime_type == "image/jpeg", f"MIME mismatch: {mime_type}"
    assert data_uri.startswith("data:image/jpeg;base64,"), f"Invalid prefix: {data_uri[:30]}"
    assert data_uri == f"data:image/jpeg;base64,{raw_b64}", "data_uri does not match raw_b64"
    
    # Decode raw base64 and verify JPEG binary format
    raw_bytes = base64.b64decode(raw_b64)
    assert raw_bytes[:2] == b'\xff\xd8', "JPEG header magic bytes mismatch"
    assert raw_bytes[-2:] == b'\xff\xd9', "JPEG EOF bytes mismatch"
    
    decoded_img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded_img is not None, "Failed to decode generated JPEG"
    assert decoded_img.shape == (480, 640, 3), f"Shape mismatch: {decoded_img.shape}"
    print("[PASS] generate_annotated_frame verified successfully with valid JPEG and Base64 output!")

def test_websocket_thread_safety():
    print("Testing WebSocketBroadcaster thread safety...")
    from app import WebSocketBroadcaster
    
    class MockWebSocket:
        def __init__(self):
            self.received = []
            self.lock = threading.Lock()
            self.closed = False
            
        def send(self, data):
            if self.closed:
                raise RuntimeError("Socket closed")
            # Verify data is valid UTF-8 string and valid JSON
            assert isinstance(data, str), "Sent data must be a string text frame"
            parsed = json.loads(data)
            assert "type" in parsed
            with self.lock:
                self.received.append(parsed)

    broadcaster = WebSocketBroadcaster()
    mock_ws1 = MockWebSocket()
    mock_ws2 = MockWebSocket()
    broadcaster.register(mock_ws1)
    broadcaster.register(mock_ws2)
    
    # Concurrently send messages from 10 threads
    threads = []
    def worker(tid):
        for i in range(20):
            payload = {
                "type": "detection_update",
                "thread_id": tid,
                "msg_id": i,
                "timestamp": time.time(),
                "test_field": "test_value"
            }
            if tid % 2 == 0:
                broadcaster.broadcast(payload)
            else:
                broadcaster.send(mock_ws1, payload)

    for tid in range(10):
        t = threading.Thread(target=worker, args=(tid,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    assert len(mock_ws1.received) > 0, "mock_ws1 received no messages"
    assert len(mock_ws2.received) > 0, "mock_ws2 received no messages"
    print(f"[PASS] WebSocketBroadcaster sent {len(mock_ws1.received)} messages concurrently with 0 UTF-8 / JSON framing errors!")

def test_plate_refinement():
    print("Testing refine_indonesian_plate...")
    from app import refine_indonesian_plate
    
    # Check that '225 BK' is not converted to 'Z 225 BK'
    res = refine_indonesian_plate("225 BK")
    print(f"refine_indonesian_plate('225 BK') -> '{res}'")
    assert "Z 225 BK" not in res and res == "", f"Expected empty / no 'Z' prefix hallucination, got '{res}'"
    
    # Check standard plate
    res_b = refine_indonesian_plate("B 1958 RZH")
    print(f"refine_indonesian_plate('B 1958 RZH') -> '{res_b}'")
    assert res_b == "B 1958 RZH", f"Expected B 1958 RZH, got {res_b}"
    
    # Check smart disambiguation & partition search cases
    cases = [
        ("B14S2HOS", "B 1452 HOS"),
        ("B141300F", "B 1413 OOF"),
        ("B2437BR5", "B 2437 BRS"),
        ("DK1234AB", "DK 1234 AB"),
        ("82489PZH", "B 2489 PZH"),
        ("B 2049 88K", "B 2049 BBK"),
    ]
    for inp, expected in cases:
        out = refine_indonesian_plate(inp)
        print(f"refine_indonesian_plate('{inp}') -> '{out}'")
        assert out == expected, f"Failed for '{inp}': expected '{expected}', got '{out}'"
    print("[PASS] refine_indonesian_plate verified successfully with all disambiguation cases!")

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

def test_plate_bbox_in_snapshot():
    print("Testing plate_bbox resolution in save_parking_record & confirmation...")
    from app import VehicleConfirmationManager, save_parking_record, confirmation_manager
    import time
    
    # Test 1: Track confirmation preserves last_plate_bbox even if best_f had None
    tid, trk = confirmation_manager.get_or_create_track(99, [100, 100, 400, 400], 1920, 1080, frame_id=1)
    trk.last_plate_bbox = [180, 250, 320, 300]
    trk.add_frame({
        "vehicle_type": "car",
        "vehicle_confidence": 0.95,
        "body_style": "MPV",
        "body_style_confidence": 0.90,
        "bbox": [100, 100, 400, 400],
        "plate_bbox": None,
        "inside_interest_area": True
    })
    trk._finalize_confirmation("B 1452 HOS", 0.92, 2, reason="TEST_CONFIRM")
    assert trk.confirmed_data is not None, "confirmed_data is None"
    assert trk.confirmed_data.get("plate_bbox") == [180, 250, 320, 300], f"Expected plate_bbox resolved, got: {trk.confirmed_data.get('plate_bbox')}"
    
    # Test 2: save_parking_record fills plate_bbox from track if missing
    dummy_det = {
        "track_id": 99,
        "license_plate": "B 1452 HOS",
        "plate_confidence": 0.92,
        "status": "CONFIRMED",
        "bbox": [100, 100, 400, 400],
        "plate_bbox": None
    }
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rec = save_parking_record(dummy_det, source_img=dummy_frame)
    assert dummy_det.get("plate_bbox") == [180, 250, 320, 300], f"Expected save_parking_record to populate plate_bbox, got {dummy_det.get('plate_bbox')}"
    print("[PASS] Plate bbox resolution in snapshot verified successfully!")

if __name__ == "__main__":
    test_annotated_image()
    test_websocket_thread_safety()
    test_plate_refinement()
    test_track_isolation()
    test_plate_bbox_in_snapshot()
    print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
