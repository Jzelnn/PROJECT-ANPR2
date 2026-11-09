#!/bin/bash
# ========================================================
# Skrip Menjalankan ANPR Parking System di Linux / Raspberry Pi
# ========================================================
export OPENBLAS_CORETYPE=ARMV8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DISABLE_PADDLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

python backend/app.py
