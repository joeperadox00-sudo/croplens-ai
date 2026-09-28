"""
CROPLENS AI: Quantization Module
Slide 2 & 7: Quantizing 32-bit floating point weights into 8-bit integers (INT8)
for ultra-fast, low-memory CPU execution without GPU dependency.
100% Offline, no external APIs.
"""

import io
import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


def quantize_model_dynamic(model: nn.Module) -> nn.Module:
    """
    Applies PyTorch Dynamic Quantization (FP32 -> INT8) on linear layers.
    Optimized for CPU inference on low-resource devices (Slide 4).
    """
    # Create deep copy so original is unaffected
    model_cpu = copy.deepcopy(model).cpu()

    # Dynamic quantization for linear layers
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model


def get_model_size_bytes(model: nn.Module) -> int:
    """
    Calculates the exact state_dict storage size in bytes.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def compare_quantization_stats(original_model: nn.Module, quantized_model: nn.Module) -> Dict[str, Any]:
    """
    Compares size and weight precision between FP32 and INT8 models.
    """
    fp32_size = get_model_size_bytes(original_model)
    int8_size = get_model_size_bytes(quantized_model)
    reduction = ((fp32_size - int8_size) / fp32_size * 100.0) if fp32_size > 0 else 0.0

    return {
        "fp32_size_bytes": fp32_size,
        "fp32_size_kb": round(fp32_size / 1024, 3),
        "int8_size_bytes": int8_size,
        "int8_size_kb": round(int8_size / 1024, 3),
        "size_reduction_percent": round(reduction, 1),
        "precision_original": "FP32 (32-bit float)",
        "precision_quantized": "INT8 (8-bit integer)"
    }
