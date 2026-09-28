"""
CROPLENS AI - Model Pruning Script
Prunes redundant parameters and measures sparsity.
100% Offline, no external APIs.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.models import CropLensMicroNet
from src.pruning import apply_l1_pruning, calculate_sparsity


def run_pruning(amount: float = 0.3):
    print("==================================================")
    print(f" CROPLENS AI: WEIGHT PRUNING ({int(amount*100)}% Sparsity) ")
    print("==================================================")

    model = CropLensMicroNet()
    ckpt_path = os.path.join("models", "croplens_micronet.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        print(f"[OK] Loaded checkpoint from {ckpt_path}")

    print(f"Before Pruning: {model.count_parameters()} total parameters")

    pruned_model = apply_l1_pruning(model, amount=amount)
    stats = calculate_sparsity(pruned_model)

    print("\n--- Pruning Results ---")
    print(f"Total Weights:          {stats['total_weights']}")
    print(f"Zeroed Weights:         {stats['zero_weights']}")
    print(f"Active Weights:         {stats['active_weights']}")
    print(f"Sparsity Achieved:      {stats['sparsity_percent']}%")
    print(f"Active Model Parameters: {stats['active_model_parameters']} (Original: {stats['total_model_parameters']})")

    out_path = os.path.join("models", "croplens_micronet_pruned.pt")
    torch.save(pruned_model.state_dict(), out_path)
    print(f"[SUCCESS] Pruned model saved to {out_path}")


if __name__ == "__main__":
    run_pruning(amount=0.3)
