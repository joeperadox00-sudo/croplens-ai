"""
CROPLENS AI - Post-Training Quantization Script
Converts FP32 weights to INT8 precision for low-power edge CPU inference.
100% Offline, no external APIs.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.models import CropLensMicroNet
from src.quantization import quantize_model_dynamic, compare_quantization_stats


def run_quantization():
    print("==================================================")
    print(" CROPLENS AI: POST-TRAINING INT8 QUANTIZATION ")
    print("==================================================")

    model = CropLensMicroNet()
    ckpt_path = os.path.join("models", "croplens_micronet.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        print(f"[OK] Loaded checkpoint from {ckpt_path}")

    quantized_model = quantize_model_dynamic(model)
    stats = compare_quantization_stats(model, quantized_model)

    print("\n--- Quantization Metrics ---")
    print(f"FP32 Size:            {stats['fp32_size_bytes']} bytes ({stats['fp32_size_kb']} KB)")
    print(f"INT8 Quantized Size:  {stats['int8_size_bytes']} bytes ({stats['int8_size_kb']} KB)")
    print(f"Storage Reduction:    {stats['size_reduction_percent']}%")
    print(f"Original Precision:   {stats['precision_original']}")
    print(f"Optimized Precision:  {stats['precision_quantized']}")

    out_path = os.path.join("models", "croplens_micronet_int8.pt")
    torch.save(quantized_model.state_dict(), out_path)
    print(f"[SUCCESS] Quantized INT8 model saved to {out_path}")


if __name__ == "__main__":
    run_quantization()
