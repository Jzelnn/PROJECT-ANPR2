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

def test_plate_bbox_projection_and_stale_rejection():
    print("Testing plate_bbox relative projection and stale frame_id rejection...")
    from app import VehicleTrack

    trk = VehicleTrack(77, initial_det={"inside_interest_area": True, "bbox": [267, 163, 660, 484]})
    trk.last_bbox = [267, 163, 660, 484]
    # 1. Before any plate detection, get_current_plate_bbox returns None
    assert trk.get_current_plate_bbox([267, 163, 660, 484]) is None

    # 2. Plate detected on frame 14: [477, 377, 561, 422] inside vehicle [267, 163, 660, 484]
    ok14 = trk.update_plate_bbox([477, 377, 561, 422], [267, 163, 660, 484], frame_id=14)
    assert ok14 is True
    assert trk.last_plate_frame_id == 14
    assert trk.rel_plate_bbox is not None

    # 3. Vehicle moves forward on frame 16 to [280, 164, 709, 517]
    proj16 = trk.get_current_plate_bbox([280, 164, 709, 517])
    assert proj16 is not None
    # Since vehicle moved right (+13..+49 px) and down (+1..+33 px), projected plate must move right and down too!
    assert proj16[0] > 477 and proj16[1] > 377, f"Expected projected plate to follow vehicle movement, got {proj16}"

    # 4. Newer plate detection on frame 16 updates the track
    ok16 = trk.update_plate_bbox([505, 412, 598, 460], [280, 164, 709, 517], frame_id=16)
    assert ok16 is True
    assert trk.last_plate_frame_id == 16

    # 5. Late/stale plate detection from frame 15 arrives out-of-order -> MUST BE REJECTED
    ok15_stale = trk.update_plate_bbox([497, 405, 591, 455], [279, 167, 695, 508], frame_id=15)
    assert ok15_stale is False, "Expected stale frame 15 plate update to be rejected after frame 16"
    assert trk.last_plate_frame_id == 16
    assert trk.last_plate_bbox == [505, 412, 598, 460], "Stale frame 15 overwrote frame 16 plate_bbox!"

    # 6. Fallback when rel_plate_bbox is None -> returns None when moving vehicle given to prevent stale freezing
    trk.rel_plate_bbox = None
    assert trk.get_current_plate_bbox([300, 180, 720, 530]) is None, "Expected None instead of frozen stale coordinates for moving vehicle"
    assert trk.get_current_plate_bbox(None) == [505, 412, 598, 460], "Expected last_plate_bbox when cur_vbbox is None and fresh"
    print("[PASS] Plate bbox relative projection and stale frame_id rejection verified successfully!")

def test_entry_history_body_style_flow():
    print("Testing Entry History body_style flow (pending state & async body_update resolution)...")
    from app import confirmation_manager, save_parking_record, _async_body_task, LATEST_RECORDS, ws_broadcaster

    class MockWS:
        def __init__(self):
            self.messages = []
        def send(self, data):
            self.messages.append(json.loads(data))

    mock_ws = MockWS()
    ws_broadcaster.register(mock_ws)
    try:
        tid, trk = confirmation_manager.get_or_create_track(55, [267, 163, 660, 484], 1280, 674, frame_id=14)
        trk.body_pending = True
        trk.last_plate_bbox = [477, 377, 561, 422]
        trk.add_frame({
            "frame_id": 14,
            "vehicle_type": "car",
            "vehicle_confidence": 0.94,
            "body_style": None,
            "body_style_confidence": 0.0,
            "bbox": [267, 163, 660, 484],
            "plate_bbox": [477, 377, 561, 422],
            "inside_interest_area": True
        })
        # 1. Confirm plate BEFORE async body task finishes
        trk._finalize_confirmation("G 2360 GRS", 0.88, 1, reason="FAST_STRONG_STRUCTURE")
        assert trk.confirmed_data["body_style"] is None
        assert trk.confirmed_data["body_pending"] is True

        dummy_img = np.zeros((674, 1280, 3), dtype=np.uint8)
        rec = save_parking_record(dict(trk.confirmed_data), source_img=dummy_img)
        assert rec is not None
        assert rec["body_style"] is None
        assert rec["body_pending"] is True

        # 2. Simulate track starting to exit ROI before _async_body_task finishes
        trk.lifecycle_state = "EXITING"

        # 3. Run _async_body_task with a strong Sedan hint -> must update confirmed_data, LATEST_RECORDS, and broadcast body_update with record_id
        _async_body_task(
            55, dummy_img, [267, 163, 660, 484], "car", 0.94, False,
            body_hints=[{"name": "Sedan", "conf": 0.91, "box": [267, 163, 660, 484]}],
            frame_id=14, submit_time=time.time()
        )
        assert trk.cached_body_style == "Sedan", f"Expected Sedan, got {trk.cached_body_style}"
        assert trk.confirmed_data["body_style"] == "Sedan", f"Expected confirmed_data body_style='Sedan', got {trk.confirmed_data['body_style']}"
        assert trk.confirmed_data["body_pending"] is False

        saved_rec = next((r for r in LATEST_RECORDS if r.get("id") == rec["id"]), None)
        assert saved_rec is not None
        assert saved_rec["body_style"] == "Sedan", f"Expected LATEST_RECORDS body_style='Sedan', got {saved_rec['body_style']}"
        assert saved_rec["body_pending"] is False

        body_msgs = [m for m in mock_ws.messages if m.get("type") == "body_update" and m.get("track_id") == 55]
        assert len(body_msgs) == 1, f"Expected 1 body_update WS message, got {len(body_msgs)}"
        assert body_msgs[0]["body_style"] == "Sedan"
        assert body_msgs[0]["record_id"] == rec["id"]
        print("[PASS] Entry History body_style flow verified successfully!")
    finally:
        ws_broadcaster.unregister(mock_ws)

def test_flicker_tolerance_in_async_ocr():
    print("Testing 1-frame tracking flicker tolerance for async OCR results...")
    from app import VehicleConfirmationManager, confirmation_manager
    
    # 1. Vehicle enters and is active
    tid, trk = confirmation_manager.get_or_create_track(88, [100, 100, 400, 400], 1920, 1080, frame_id=10)
    trk.lifecycle_state = "ACTIVE"
    trk.frames_missing = 0
    trk.ever_inside_roi = True
    trk.inside_interest_area = True

    # 2. Simulate 1-frame tracking miss (flicker)
    confirmation_manager.update([], is_stream=True)
    assert trk.frames_missing == 1
    assert trk.lifecycle_state == "ACTIVE", f"Expected ACTIVE during 1-frame flicker, got {trk.lifecycle_state}"

    # 3. Add OCR result while vehicle is in 1-frame flicker -> must NOT be discarded
    update_info = trk.add_ocr_result(job_id=1, text="B 2162 BRK", conf=0.88, crop_q=1.2, ocr_method="test")
    assert update_info is not None, "Expected OCR result accepted during 1-frame flicker"
    assert trk.best_candidate == "B2162BRK"

    # 4. Simulate > 3 frames missing -> now it should transition to EXITING
    for _ in range(4):
        confirmation_manager.update([], is_stream=True)
    assert trk.frames_missing >= 4
    assert trk.lifecycle_state == "EXITING", f"Expected EXITING after >3 frames missing, got {trk.lifecycle_state}"
    print("[PASS] 1-frame tracking flicker tolerance verified successfully!")

def test_periodic_plate_alignment_when_locked():
    print("Testing periodic plate alignment for locked tracks without triggering OCR...")
    from app import confirmation_manager, _async_plate_task, ocr_executor

    confirmation_manager.reset()
    tid, trk = confirmation_manager.get_or_create_track(88, [200, 200, 600, 500], 1920, 1080, frame_id=1)
    trk.is_locked = True
    trk.status = "CONFIRMED"
    trk.best_candidate = "B1234ABC"
    trk.best_candidate_display = "B 1234 ABC"
    trk.confirmed_data = {"license_plate": "B 1234 ABC", "plate_bbox": [300, 400, 500, 450]}

    # Dummy full image with vehicle crop
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    dummy_crop = dummy_img[200:500, 200:600]

    # Run _async_plate_task with alignment_only=True
    initial_ocr_pending = trk.ocr_pending
    _async_plate_task(
        tid,
        dummy_crop,
        [200, 200, 600, 500],
        dummy_img,
        1920,
        1080,
        time.time(),
        initial_vtype="car",
        frame_id=2,
        alignment_only=True
    )

    # Verify that OCR was NOT queued and no pending OCR job was created
    assert trk.ocr_pending == initial_ocr_pending, "alignment_only must NOT set ocr_pending"
    assert trk.pending_ocr_job is None, "alignment_only must NOT queue pending_ocr_job"
    assert trk.is_locked is True, "Track lock must remain intact"
    assert trk.best_candidate == "B1234ABC", "OCR candidate must not be modified by alignment task"
    print("[PASS] Periodic plate alignment without OCR verified successfully!")

def test_corroborating_ocr_confirmation():
    print("Testing corroborating OCR reading with slightly lower confidence...")
    from app import VehicleTrack

    trk = VehicleTrack(101, initial_det={
        "inside_interest_area": True,
        "bbox": [200, 150, 600, 480],
        "vehicle_type": "car",
        "vehicle_confidence": 0.90,
        "body_style": "MPV",
        "body_style_confidence": 0.85
    })

    # First read: conf=0.63 (<0.68 single-frame fast confirm, so needs 2 reads), crop_q=1.20
    info1 = trk.add_ocr_result(job_id=1, text="B 1452 HOS", conf=0.63, crop_q=1.20)
    assert info1 is not None
    assert trk.status == "ANALYZING"
    assert trk.candidate_counts.get("B1452HOS") == 1
    assert trk.best_candidate_conf == 0.63

    # Second read of the SAME plate: conf=0.52 (which is < 0.63 - 0.08 = 0.55) and crop_q=1.05 (< 1.20)
    # Previously this was rejected by the weaker-result guard; now it must corroborate and confirm!
    info2 = trk.add_ocr_result(job_id=2, text="B 1452 HOS", conf=0.52, crop_q=1.05)
    assert info2 is not None
    assert trk.candidate_counts.get("B1452HOS") == 2, f"Expected candidate_counts=2, got {trk.candidate_counts.get('B1452HOS')}"
    assert trk.best_candidate_conf == 0.63, f"Expected highest confidence 0.63 preserved, got {trk.best_candidate_conf}"
    assert trk.status == "CONFIRMED", f"Expected CONFIRMED after 2 corroborating reads, got {trk.status}"
    assert info2["ocr_confidence"] == 0.63

    # Verify a DIFFERENT weaker plate is still rejected on an unconfirmed track
    trk2 = VehicleTrack(102)
    trk2.add_ocr_result(job_id=1, text="B 1413 DOF", conf=0.64, crop_q=1.30)
    info_diff = trk2.add_ocr_result(job_id=2, text="B 9999 XYZ", conf=0.50, crop_q=1.00)
    assert trk2.best_candidate == "B1413DOF", "Different weaker plate must not overwrite best_candidate"
    assert trk2.candidate_counts.get("B9999XYZ", 0) == 0, "Different weaker plate must still be rejected"
    print("[PASS] Corroborating OCR reading confirmation & highest confidence preservation verified!")

def test_ocr_fallback_preemption_and_pending_crop_guard():
    print("Testing _should_skip_ocr_fallbacks and ensemble_plate_reading preemption...")
    from app import confirmation_manager, _should_skip_ocr_fallbacks, ensemble_plate_reading

    confirmation_manager.reset()
    tid, trk = confirmation_manager.get_or_create_track(105, [200, 200, 600, 500], 1920, 1080, frame_id=10)
    trk.lifecycle_state = "ACTIVE"

    # 1. No pending_ocr_job -> should NOT skip fallbacks
    assert _should_skip_ocr_fallbacks(tid, 1.0) is False

    # 2. Worse crop queued (crop_q=0.8 < current 1.0) -> should NOT skip fallbacks
    trk.pending_ocr_job = (2, tid, np.zeros((40, 120, 3), dtype=np.uint8), 0.80, None, time.time(), 11, time.time())
    assert _should_skip_ocr_fallbacks(tid, 1.0) is False

    # 3. Equally good or better crop queued (crop_q=1.25 >= current 1.0) -> SHOULD skip fallbacks
    trk.pending_ocr_job = (3, tid, np.zeros((40, 120, 3), dtype=np.uint8), 1.25, None, time.time(), 12, time.time())
    assert _should_skip_ocr_fallbacks(tid, 1.0) is True

    # 4. Verify ensemble_plate_reading returns fast_preempted_by_better_crop when callback returns True
    blank_crop = np.zeros((45, 140, 3), dtype=np.uint8)
    res = ensemble_plate_reading(blank_crop, should_skip_fallbacks=lambda: True)
    assert res.get("method") == "fast_preempted_by_better_crop", f"Expected fast_preempted_by_better_crop, got {res.get('method')}"
    print("[PASS] OCR fallback preemption and pending crop quality guard verified!")

if __name__ == "__main__":
    import app
    orig_history = list(app.LATEST_RECORDS)
    orig_file_bytes = None
    if os.path.exists(app.HISTORY_FILE):
        with open(app.HISTORY_FILE, "rb") as f:
            orig_file_bytes = f.read()
    try:
        test_annotated_image()
        test_websocket_thread_safety()
        test_plate_refinement()
        test_track_isolation()
        test_plate_bbox_in_snapshot()
        test_plate_bbox_projection_and_stale_rejection()
        test_entry_history_body_style_flow()
        test_flicker_tolerance_in_async_ocr()
        test_periodic_plate_alignment_when_locked()
        test_corroborating_ocr_confirmation()
        test_ocr_fallback_preemption_and_pending_crop_guard()
        print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    finally:
        app.LATEST_RECORDS.clear()
        app.LATEST_RECORDS.extend(orig_history)
        if orig_file_bytes is not None:
            with open(app.HISTORY_FILE, "wb") as f:
                f.write(orig_file_bytes)
