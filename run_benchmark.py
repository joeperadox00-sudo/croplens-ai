"""
CROPLENS AI - Standalone Comparative Benchmark Script
Evaluates Heavy Baseline vs CROPLENS AI (<30 params) across Latency, Size, Params, and RAM.
100% Offline, no external APIs.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.benchmark import run_comprehensive_benchmark


def main():
    print("Running CROPLENS AI comparative benchmark on local CPU...")
    results = run_comprehensive_benchmark()

    print("\n" + "=" * 65)
    print("          CROPLENS AI: EDGE PERFORMANCE BENCHMARK          ")
    print("=" * 65)

    teacher = results["teacher"]
    croplens = results["croplens_optimized"]
    comp = results["comparison_highlights"]

    teacher_size_str = f"{teacher['size_mb']} MB"
    croplens_size_str = f"{croplens['size_kb']} KB"
    teacher_lat_mean = f"{teacher['latency']['mean_ms']:.2f} ms"
    croplens_lat_mean = f"{croplens['latency']['mean_ms']:.3f} ms"
    teacher_lat_p95 = f"{teacher['latency']['p95_ms']:.2f} ms"
    croplens_lat_p95 = f"{croplens['latency']['p95_ms']:.3f} ms"

    print(f"{'Metric':<25} | {'Heavy Baseline (Slide 6)':<22} | {'CROPLENS AI (Slide 7)':<22}")
    print("-" * 75)
    print(f"{'Model Architecture':<25} | {'Deep CNN (ResNet)':<22} | {'MicroNet (Bio-Optical)':<22}")
    print(f"{'Parameter Count':<25} | {teacher['parameters_str']:<22} | {croplens['parameters_str']:<22}")
    print(f"{'Model Disk Size':<25} | {teacher_size_str:<22} | {croplens_size_str:<22}")
    print(f"{'CPU Latency (mean)':<25} | {teacher_lat_mean:<22} | {croplens_lat_mean:<22}")
    print(f"{'CPU Latency (p95)':<25} | {teacher_lat_p95:<22} | {croplens_lat_p95:<22}")
    print(f"{'GPU Dependency':<25} | {'Required/Advised':<22} | {'Zero (CPU Native)':<22}")
    print(f"{'Edge Microcontroller':<25} | {'Incompatible':<22} | {'Fully Deployable':<22}")
    print("-" * 75)
    print(f"Summary: {comp['parameter_reduction_percent']}% Parameter Reduction | {comp['cpu_speedup_factor']}x Faster on CPU")
    print(f"Deployment: {comp['offline_status']}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
