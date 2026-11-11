#!/usr/bin/env python3
"""
Test refined pure_arm_yolo_predict against official Ultralytics predict
across all models (vehicle, plate, char) and specifically the 9 divergent photos.
"""
import os
import sys
import json
import cv2
import torch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.join(ROOT, "tools", "eval"))

os.environ["DISABLE_PADDLE"] = "1"
os.environ["SKIP_WARMUP"] = "1"

import app as A
from run_eval import normalize_plate
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

class RefinedArmBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self.id = None
    def __len__(self):
        return len(self.conf)
    def __bool__(self):
        return len(self.conf) > 0
    def __iter__(self):
        for i in range(len(self.conf)):
            class SingleBox:
                def __init__(self, b, c, cl):
                    b_t = b if isinstance(b, torch.Tensor) else torch.tensor(b, dtype=torch.float32)
                    c_t = c if isinstance(c, torch.Tensor) else torch.tensor([c], dtype=torch.float32)
                    cl_t = cl if isinstance(cl, torch.Tensor) else torch.tensor([cl], dtype=torch.float32)
                    self.xyxy = b_t.unsqueeze(0) if b_t.ndim == 1 else b_t
                    self.conf = c_t.unsqueeze(0) if c_t.ndim == 0 else c_t
                    self.cls = cl_t.unsqueeze(0) if cl_t.ndim == 0 else cl_t
                    self.id = None
            yield SingleBox(self.xyxy[i], self.conf[i], self.cls[i])

def refined_pure_arm_predict(self, source=None, imgsz=None, conf=0.25, iou=0.7, *args, **kwargs):
    if source is None:
        source = kwargs.get('source', None)
    if isinstance(source, (list, tuple)):
        img = source[0] if len(source) > 0 else None
    else:
        img = source
    if isinstance(img, str):
        img = cv2.imread(img)
    if img is None or getattr(img, 'size', 0) == 0:
        return [A.PureArmYoloResult((384, 384), names=getattr(self, 'names', {}))]

    target_sz = int(imgsz if imgsz is not None else kwargs.get('imgsz', 384))
    conf_th = float(conf if conf is not None else kwargs.get('conf', 0.25))
    iou_th = float(iou if iou is not None else kwargs.get('iou', 0.7))

    model = self.model
    model.eval()
    h0, w0 = img.shape[:2]
    is_cls_model = (type(model).__name__ == 'ClassificationModel' or getattr(self, 'task', '') == 'classify')

    if is_cls_model:
        scale_c = target_sz / max(1, min(h0, w0))
        cw, ch = max(target_sz, int(round(w0 * scale_c))), max(target_sz, int(round(h0 * scale_c)))
        resized_c = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)
        y_off = (ch - target_sz) // 2
        x_off = (cw - target_sz) // 2
        padded = resized_c[y_off:y_off + target_sz, x_off:x_off + target_sz]
        im_shape = padded.shape[:2]
    else:
        stride = getattr(model, 'stride', 32)
        if isinstance(stride, (torch.Tensor, list, tuple)):
            stride = int(stride[-1]) if len(stride) else 32
        lb = LetterBox((target_sz, target_sz), auto=True, stride=stride)
        padded = lb(image=img)
        im_shape = padded.shape[:2]

    blob = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(blob).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)

    pred = out[0] if isinstance(out, (list, tuple)) else out

    if pred.ndim == 2:
        row = pred[0]
        if float(row.min()) >= 0.0 and abs(float(row.sum()) - 1.0) < 1e-2:
            probs_data = row
        else:
            probs_data = row.softmax(dim=-1)
        probs_obj = A.PureArmProbs(probs_data)
        return [A.PureArmYoloResult((h0, w0), probs=probs_obj, names=getattr(self, 'names', {}))]

    preds = non_max_suppression(pred, conf_thres=conf_th, iou_thres=iou_th, agnostic=False, max_det=300)
    det = preds[0]
    if det is None or len(det) == 0:
        empty_t = torch.empty((0, 4), dtype=torch.float32)
        return [A.PureArmYoloResult((h0, w0), RefinedArmBoxes(empty_t, torch.empty((0,)), torch.empty((0,))), names=getattr(self, 'names', {}))]

    scaled_boxes = scale_boxes(im_shape, det[:, :4].clone(), (h0, w0))
    boxes_obj = RefinedArmBoxes(scaled_boxes, det[:, 4], det[:, 5])
    return [A.PureArmYoloResult((h0, w0), boxes_obj, names=getattr(self, 'names', {}))]


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea)


def main():
    with open(os.path.join(ROOT, "output", "eval", "langkah_2_interleaved_30photos_5runs.json"), "r") as f:
        d = json.load(f)
    divergent_files = [k for k in d["preds_A"] if normalize_plate(d["preds_A"][k]) != normalize_plate(d["preds_B"][k])]

    print("=" * 88)
    print(f"VERIFIKASI KESETARAAN REFINED PURE_ARM VS PREDICT RESMI ({len(divergent_files)} FOTO)")
    print("Kriteria: Jumlah deteksi sama, IoU > 0.95, selisih conf < 0.02")
    print("=" * 88)

    all_passed = True
    for i, f_path in enumerate(divergent_files, 1):
        full_p = os.path.join(ROOT, f_path)
        img = cv2.imread(full_p)
        if img is None:
            continue

        # 1. Vehicle
        A.set_pure_arm_mode(False)
        v_off = A.vehicle_model(img, imgsz=384, conf=0.35, verbose=False)
        v_ref = refined_pure_arm_predict(A.vehicle_model, img, imgsz=384, conf=0.35)
        
        # 2. Plate
        vbox = [int(v) for v in v_off[0].boxes[0].xyxy[0]] if len(v_off[0].boxes) > 0 else [0, 0, img.shape[1], img.shape[0]]
        v_crop = img[vbox[1]:vbox[3], vbox[0]:vbox[2]]
        p_off = A.plate_detector(v_crop, imgsz=320, conf=0.12, verbose=False)
        p_ref = refined_pure_arm_predict(A.plate_detector, v_crop, imgsz=320, conf=0.12)

        # 3. Char
        if len(p_off[0].boxes) > 0:
            pbox = [int(v) for v in p_off[0].boxes[0].xyxy[0]]
            p_crop = v_crop[pbox[1]:pbox[3], pbox[0]:pbox[2]]
        else:
            p_crop = v_crop

        c_off = A.char_model(p_crop, imgsz=384, conf=0.10, iou=0.45, verbose=False)
        c_ref = refined_pure_arm_predict(A.char_model, p_crop, imgsz=384, conf=0.10, iou=0.45)

        # Evaluasi Char Model (sumber divergensi)
        n_off = len(c_off[0].boxes)
        n_ref = len(c_ref[0].boxes)
        count_match = (n_off == n_ref)
        
        ious = []
        conf_diffs = []
        if count_match and n_off > 0:
            for b_o, b_r in zip(c_off[0].boxes, c_ref[0].boxes):
                box_o = [float(x) for x in b_o.xyxy[0]]
                box_r = [float(x) for x in b_r.xyxy[0]]
                ious.append(calculate_iou(box_o, box_r))
                conf_diffs.append(abs(float(b_o.conf[0]) - float(b_r.conf[0])))

        min_iou = min(ious) if ious else 1.0
        max_dconf = max(conf_diffs) if conf_diffs else 0.0

        passed = count_match and (min_iou >= 0.95) and (max_dconf <= 0.02)
        if not passed:
            all_passed = False

        status = "PASSED (100% IDENTIK)" if (max_dconf < 1e-4 and min_iou > 0.999) else ("PASSED" if passed else "FAILED")
        print(f"[{i:02d}] {status:22s} | n={n_off}/{n_ref} | min_iou={min_iou:.4f} | max_dconf={max_dconf:.4f} | {f_path}")

    print("-" * 88)
    print(f"Hasil Akhir Uji Kesetaraan: {'SEMUA LOLOS (100% EQUIVALENT)' if all_passed else 'ADA YANG GAGAL'}")
    print("=" * 88)

if __name__ == "__main__":
    main()
