#!/usr/bin/env python3
"""
Diagnose divergence between pure_arm_yolo_predict and official Ultralytics predict
per model (vehicle, plate, char) across the 9 divergent photos.
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

# Load 30 photos JSON
with open(os.path.join(ROOT, "output", "eval", "langkah_2_interleaved_30photos_5runs.json"), "r") as f:
    eval_data = json.load(f)

preds_a = eval_data["preds_A"]
preds_b = eval_data["preds_B"]

divergent_files = [f for f in preds_a if normalize_plate(preds_a[f]) != normalize_plate(preds_b[f])]

print(f"Total divergent photos: {len(divergent_files)}")
for i, f in enumerate(divergent_files, 1):
    print(f"[{i}] {f}")
    print(f"     Variant A (pure_arm): {preds_a[f]!r}")
    print(f"     Variant B (official): {preds_b[f]!r}")

print("\n" + "="*80)
print("ANALYZING PER-MODEL DIVERGENCE (VEHICLE -> PLATE -> CHAR)")
print("="*80)

def compare_boxes(name, res_arm, res_off):
    boxes_arm = res_arm[0].boxes
    boxes_off = res_off[0].boxes
    
    n_arm = len(boxes_arm)
    n_off = len(boxes_off)
    
    arm_confs = [round(float(b.conf[0]), 4) for b in boxes_arm]
    off_confs = [round(float(b.conf[0]), 4) for b in boxes_off]
    arm_cls = [int(b.cls[0]) for b in boxes_arm]
    off_cls = [int(b.cls[0]) for b in boxes_off]
    
    arm_xyxy = [[round(float(v), 2) for v in b.xyxy[0]] for b in boxes_arm]
    off_xyxy = [[round(float(v), 2) for v in b.xyxy[0]] for b in boxes_off]
    
    diff = (n_arm != n_off) or (arm_cls != off_cls) or (max([abs(a-b) for a, b in zip(arm_confs, off_confs)] or [0]) > 0.02)
    return {
        "model": name,
        "n_arm": n_arm, "n_off": n_off,
        "arm_cls": arm_cls, "off_cls": off_cls,
        "arm_confs": arm_confs, "off_confs": off_confs,
        "arm_xyxy": arm_xyxy, "off_xyxy": off_xyxy,
        "divergent": diff
    }

for idx, rel_path in enumerate(divergent_files, 1):
    full_path = os.path.join(ROOT, rel_path)
    if not os.path.exists(full_path):
        continue
    img = cv2.imread(full_path)
    if img is None:
        continue
        
    print(f"\n--- Photo #{idx}: {rel_path} ---")
    
    # 1. Vehicle Model
    # A: pure_arm
    A.set_pure_arm_mode(True)
    res_veh_arm = A.vehicle_model(img, imgsz=A.IMG_SIZE, conf=A.VEHICLE_CONF_THRESH, verbose=False)
    # B: official
    A.set_pure_arm_mode(False)
    res_veh_off = A.vehicle_model(img, imgsz=A.IMG_SIZE, conf=A.VEHICLE_CONF_THRESH, verbose=False)
    
    veh_comp = compare_boxes("Vehicle Model", res_veh_arm, res_veh_off)
    print(f"Vehicle: n_arm={veh_comp['n_arm']} vs n_off={veh_comp['n_off']} | div={veh_comp['divergent']}")
    if veh_comp["divergent"]:
        print(f"  ARM : cls={veh_comp['arm_cls']} conf={veh_comp['arm_confs']} xyxy={veh_comp['arm_xyxy']}")
        print(f"  OFF : cls={veh_comp['off_cls']} conf={veh_comp['off_confs']} xyxy={veh_comp['off_xyxy']}")
        continue

    # Crop vehicle bumper for plate model
    # Use the official detection bounding box
    if len(res_veh_off[0].boxes) > 0:
        vbox = [int(v) for v in res_veh_off[0].boxes[0].xyxy[0]]
        v_crop = img[vbox[1]:vbox[3], vbox[0]:vbox[2]]
    else:
        v_crop = img
        
    if v_crop.size == 0:
        continue
        
    # 2. Plate Model
    A.set_pure_arm_mode(True)
    res_plt_arm = A.plate_detector(v_crop, imgsz=A.PLATE_INFER_IMGSZ, conf=A.PLATE_INFER_CONF, verbose=False)
    A.set_pure_arm_mode(False)
    res_plt_off = A.plate_detector(v_crop, imgsz=A.PLATE_INFER_IMGSZ, conf=A.PLATE_INFER_CONF, verbose=False)
    
    plt_comp = compare_boxes("Plate Model", res_plt_arm, res_plt_off)
    print(f"Plate  : n_arm={plt_comp['n_arm']} vs n_off={plt_comp['n_off']} | div={plt_comp['divergent']}")
    if plt_comp["divergent"]:
        print(f"  ARM : conf={plt_comp['arm_confs']} xyxy={plt_comp['arm_xyxy']}")
        print(f"  OFF : conf={plt_comp['off_confs']} xyxy={plt_comp['off_xyxy']}")
        continue

    # Crop plate for char model
    if len(res_plt_off[0].boxes) > 0:
        pbox = [int(v) for v in res_plt_off[0].boxes[0].xyxy[0]]
        p_crop = v_crop[pbox[1]:pbox[3], pbox[0]:pbox[2]]
    else:
        p_crop = v_crop
        
    if p_crop.size == 0:
        continue

    # 3. Char Model
    A.set_pure_arm_mode(True)
    res_chr_arm = A.char_model(p_crop, imgsz=384, conf=0.10, iou=0.45, verbose=False)
    A.set_pure_arm_mode(False)
    res_chr_off = A.char_model(p_crop, imgsz=384, conf=0.10, iou=0.45, verbose=False)
    
    chr_comp = compare_boxes("Char Model", res_chr_arm, res_chr_off)
    print(f"Char   : n_arm={chr_comp['n_arm']} vs n_off={chr_comp['n_off']} | div={chr_comp['divergent']}")
    if chr_comp["divergent"]:
        # Decode characters sorted by x1
        arm_chars = [(int(b.cls[0]), float(b.conf[0]), float(b.xyxy[0][0])) for b in res_chr_arm[0].boxes]
        off_chars = [(int(b.cls[0]), float(b.conf[0]), float(b.xyxy[0][0])) for b in res_chr_off[0].boxes]
        arm_chars.sort(key=lambda x: x[2])
        off_chars.sort(key=lambda x: x[2])
        arm_text = "".join(A.char_model.names.get(c[0], str(c[0])) for c in arm_chars)
        off_text = "".join(A.char_model.names.get(c[0], str(c[0])) for c in off_chars)
        print(f"  ARM raw text: {arm_text!r} (boxes={chr_comp['n_arm']})")
        print(f"  OFF raw text: {off_text!r} (boxes={chr_comp['n_off']})")
