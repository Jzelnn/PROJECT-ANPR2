import cv2
import sys
import os

sys.path.insert(0, "backend")
import app

img_path = r"C:\Users\Lenovo\.gemini\antigravity\brain\cbb21041-1107-4e15-8424-1db246fed0e7\.user_uploaded\media_1791542680311.png"
img = cv2.imread(img_path)
if img is None:
    print("Could not load image")
    sys.exit(1)

h, w = img.shape[:2]
print(f"Image resolution: {w}x{h}")

# 1. Run vehicle detector directly
vdet = app.vehicle_model.predict(img, conf=0.15, imgsz=app.IMG_SIZE, device=app.DEVICE, verbose=False)[0]
print(f"Vehicle detector found {len(vdet.boxes)} boxes:")
for b in vdet.boxes:
    cls_id = int(b.cls[0])
    cname = app.vehicle_model.names.get(cls_id, "")
    conf = float(b.conf[0])
    xyxy = list(map(int, b.xyxy[0].tolist()))
    print(f"  Vehicle box: {cname} (cls={cls_id}) conf={conf:.2f} box={xyxy}")

# 2. Run plate detector directly on full image
pdet = app.plate_detector.predict(img, conf=0.04, imgsz=app.IMG_SIZE, device=app.DEVICE, verbose=False)[0]
print(f"Plate detector found {len(pdet.boxes)} boxes:")
for pb in pdet.boxes:
    pcls = int(pb.cls[0])
    pname = app.plate_detector.names.get(pcls, "")
    pconf = float(pb.conf[0])
    pxyxy = list(map(int, pb.xyxy[0].tolist()))
    print(f"  Plate detector box: {pname} (cls={pcls}) conf={pconf:.2f} box={pxyxy}")

# 3. Run legacy plate detector
pdet_leg = app.plate_model_legacy.predict(img, conf=0.04, device=app.DEVICE, verbose=False)[0]
print(f"Legacy plate detector found {len(pdet_leg.boxes)} boxes:")
for pb in pdet_leg.boxes:
    pcls = int(pb.cls[0])
    pname = app.plate_model_legacy.names.get(pcls, "")
    pconf = float(pb.conf[0])
    pxyxy = list(map(int, pb.xyxy[0].tolist()))
    print(f"  Legacy plate box: {pname} conf={pconf:.2f} box={pxyxy}")

# 4. Run run_anpr directly on this image
res = app.run_anpr(img, is_stream=False)
print("run_anpr detections:", res.get("detections"))
