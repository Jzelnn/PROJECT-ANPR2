import cv2
import sys
import os

sys.path.insert(0, "backend")
import app

img_path = r"C:\Users\Lenovo\.gemini\antigravity\brain\cbb21041-1107-4e15-8424-1db246fed0e7\.user_uploaded\media_1791533413395.png"
if not os.path.exists(img_path):
    print("User image not found at path:", img_path)
    # Check output/screenshots
    import glob
    shots = glob.glob("output/screenshots/*.jpg")
    if shots:
        img_path = shots[0]

print("Testing image:", img_path)
img = cv2.imread(img_path)
if img is not None:
    h, w = img.shape[:2]
    print(f"Loaded image {w}x{h}")
    # Test YOLO vehicle detection
    vdet = app.vehicle_model.predict(img, conf=0.15, imgsz=app.IMG_SIZE, device=app.DEVICE, verbose=False)[0]
    print(f"Found {len(vdet.boxes)} vehicle boxes:")
    for b in vdet.boxes:
        cls_id = int(b.cls[0])
        cname = app.vehicle_model.names.get(cls_id, "")
        conf = float(b.conf[0])
        xyxy = list(map(int, b.xyxy[0].tolist()))
        print(f"  Vehicle: {cname} conf={conf:.2f} box={xyxy}")
        
        # Test plate detection on this vehicle crop
        vx1, vy1, vx2, vy2 = xyxy
        crop = img[vy1:vy2, vx1:vx2]
        pdet = app.plate_detector.predict(crop, conf=0.05, imgsz=app.PLATE_INFER_IMGSZ, device=app.DEVICE, verbose=False)[0]
        print(f"    Plate crop detector (tight) found {len(pdet.boxes)} boxes:")
        for pb in pdet.boxes:
            pcls = int(pb.cls[0])
            pname = app.plate_detector.names.get(pcls, "")
            pconf = float(pb.conf[0])
            pxyxy = list(map(int, pb.xyxy[0].tolist()))
            print(f"      Crop plate: {pname} conf={pconf:.2f} box={pxyxy}")

        # Test context padded crop
        pad_x = int((vx2 - vx1) * 0.08)
        pad_y = int((vy2 - vy1) * 0.08)
        cx1 = max(0, vx1 - pad_x)
        cy1 = max(0, vy1 - pad_y)
        cx2 = min(w, vx2 + pad_x)
        cy2 = min(h, vy2 + pad_y)
        crop_padded = img[cy1:cy2, cx1:cx2]
        pdet_pad = app.plate_detector.predict(crop_padded, conf=0.05, imgsz=app.PLATE_INFER_IMGSZ, device=app.DEVICE, verbose=False)[0]
        print(f"    Plate crop detector (padded) found {len(pdet_pad.boxes)} boxes:")
        for pb in pdet_pad.boxes:
            pcls = int(pb.cls[0])
            pname = app.plate_detector.names.get(pcls, "")
            pconf = float(pb.conf[0])
            pxyxy = list(map(int, pb.xyxy[0].tolist()))
            abs_box = [cx1 + pxyxy[0], cy1 + pxyxy[1], cx1 + pxyxy[2], cy1 + pxyxy[3]]
            print(f"      Crop padded plate: {pname} conf={pconf:.2f} rel_box={pxyxy} abs_box={abs_box} valid={app.is_valid_plate_box(abs_box, w, h)}")
            
        # Test full image plate detection
        pdet_full = app.plate_detector.predict(img, conf=0.05, imgsz=app.IMG_SIZE, device=app.DEVICE, verbose=False)[0]
        print(f"    Plate full frame found {len(pdet_full.boxes)} boxes:")
        for pb in pdet_full.boxes:
            pcls = int(pb.cls[0])
            pname = app.plate_detector.names.get(pcls, "")
            pconf = float(pb.conf[0])
            pxyxy = list(map(int, pb.xyxy[0].tolist()))
            print(f"      Full frame plate: {pname} conf={pconf:.2f} box={pxyxy} valid={app.is_valid_plate_box(pxyxy, w, h)}")
