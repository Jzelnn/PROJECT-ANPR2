import os
import sys

os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["USE_XNNPACK"] = "0"
os.environ["TORCH_DISABLE_XNNPACK"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import numpy as np
import cv2
import torch
import torchvision

torch.set_num_threads(1)
if hasattr(torch.backends, 'mkldnn'): torch.backends.mkldnn.enabled = False
if hasattr(torch.backends, 'nnpack'): torch.backends.nnpack.enabled = False
if hasattr(torch.backends, 'xnnpack'): torch.backends.xnnpack.enabled = False

from ultralytics import YOLO

class PureArmProbs:
    def __init__(self, data):
        self.data = data
        self.top1 = int(data.argmax())
        self.top1conf = float(data[self.top1])

class PureArmBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self.id = None
    def __len__(self):
        return len(self.conf)
    def __iter__(self):
        for i in range(len(self.conf)):
            class SingleBox:
                def __init__(self, b, c, cl):
                    self.xyxy = torch.stack([b]) if b.ndim == 1 else b
                    self.conf = torch.tensor([c]) if not isinstance(c, torch.Tensor) else c
                    self.cls = torch.tensor([cl]) if not isinstance(cl, torch.Tensor) else cl
                    self.id = None
            yield SingleBox(self.xyxy[i], self.conf[i], self.cls[i])

class PureArmYoloResult:
    def __init__(self, orig_shape, boxes=None, probs=None, names=None):
        self.orig_shape = orig_shape
        self.boxes = boxes if boxes is not None else PureArmBoxes(torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,)))
        self.probs = probs
        self.names = names or {}

def pure_arm_yolo_predict(yolo_obj, img, imgsz=384, conf=0.25, iou=0.45, *args, **kwargs):
    """
    Inferensi YOLO murni (Pure PyTorch + Torchvision NMS) tanpa memanggil Ultralytics predict()
    yang memicu Illegal Instruction di Raspberry Pi Cortex-A72.
    """
    model = yolo_obj.model
    model.eval()

    h0, w0 = img.shape[:2]
    # Letterbox / resize
    r = min(imgsz / h0, imgsz / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = (imgsz - new_unpad[0]) / 2, (imgsz - new_unpad[1]) / 2
    
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR to RGB, HWC to CHW, uint8 to float32 [0, 1]
    blob = padded[:, :, ::-1].transpose(2, 0, 1)
    blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(blob).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)

    pred = out[0] if isinstance(out, (list, tuple)) else out

    # Kasus 1: Model Klasifikasi (seperti body_type model)
    if pred.ndim == 2:
        probs_data = pred[0].softmax(dim=-1)
        probs_obj = PureArmProbs(probs_data)
        return [PureArmYoloResult((h0, w0), probs=probs_obj, names=yolo_obj.names)]

    # Kasus 2: Model Deteksi Bounding Box (Vehicle, Plate, Character)
    pred = pred.transpose(1, 2)
    pred = pred[0] # (num_anchors, 4 + nc)

    boxes_cxcywh = pred[:, :4]
    scores_all = pred[:, 4:]
    max_scores, class_ids = scores_all.max(dim=-1)

    mask = max_scores > conf
    if not mask.any():
        empty_t = torch.empty((0, 4), dtype=torch.float32)
        return [PureArmYoloResult((h0, w0), PureArmBoxes(empty_t, torch.empty((0,)), torch.empty((0,))), names=yolo_obj.names)]

    boxes_cxcywh = boxes_cxcywh[mask]
    scores = max_scores[mask]
    cls = class_ids[mask]

    # Convert cx, cy, w, h to x1, y1, x2, y2 (pada padded space)
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    # Scale balik ke original image space (unpad)
    x1 = (x1 - left) / r
    y1 = (y1 - top) / r
    x2 = (x2 - left) / r
    y2 = (y2 - top) / r

    x1.clamp_(0, w0)
    y1.clamp_(0, h0)
    x2.clamp_(0, w0)
    y2.clamp_(0, h0)

    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    # NMS via torchvision.ops.nms
    keep = torchvision.ops.nms(boxes_xyxy, scores, iou)

    final_boxes = boxes_xyxy[keep]
    final_scores = scores[keep]
    final_cls = cls[keep]

    boxes_obj = PureArmBoxes(final_boxes, final_scores, final_cls)
    return [PureArmYoloResult((h0, w0), boxes_obj, names=yolo_obj.names)]


print("\n" + "=" * 55)
print("  UJI PURE ARM YOLO PREDICT DI CORTEX-A72")
print("=" * 55, flush=True)

test_list = [
    ("1. Vehicle Model", "models/vehicle/best.pt", 384),
    ("2. Plate Best Model", "models/plate/best.pt", 256),
    ("3. Plate Legacy Model", "models/plate/legacy.pt", 256),
    ("4. Body Type Model", "models/body_type/best.pt", 192),
    ("5. Character Model", "models/character/best.pt", 160)
]

for name, path, sz in test_list:
    if os.path.exists(path):
        print(f"\n[TEST] {name} ({path})...", flush=True)
        try:
            m = YOLO(path)
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            res = pure_arm_yolo_predict(m, dummy, imgsz=sz, conf=0.1)
            if res[0].probs is not None:
                top_cls = res[0].names[res[0].probs.top1]
                print(f"       ✅ {name}: SUKSES 100% (Klasifikasi: {top_cls}, Conf: {res[0].probs.top1conf:.2f})", flush=True)
            else:
                print(f"       ✅ {name}: SUKSES 100% (Deteksi Bounding Box: {len(res[0].boxes)})", flush=True)
        except Exception as e:
            print(f"       ❌ {name} GAGAL: {e}", flush=True)
            import traceback
            traceback.print_exc()

print("\n" + "=" * 55)
print("  SEMUA PENGUJIAN SELESAI!")
print("=" * 55 + "\n", flush=True)
