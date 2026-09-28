"""
CROPLENS AI: Pruning Module
Slide 2 & 7: Pruning redundant weights to compress the model and accelerate inference.
100% Offline, no external APIs.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Any, Tuple


def apply_l1_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    Applies L1-norm unstructured pruning to linear / conv layers of the model.
    amount: float between 0.0 and 0.9 (fraction of weights to prune)
    """
    pruned_model = copy.deepcopy(model)

    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            # Make pruning permanent by removing reparameterization hooks
            prune.remove(module, "weight")

    return pruned_model


def calculate_sparsity(model: nn.Module) -> Dict[str, Any]:
    """
    Calculates the exact zero-weight sparsity and non-zero parameter count.
    """
    total_weights = 0
    zero_weights = 0

    for name, param in model.named_parameters():
        if "weight" in name:
            weights = param.data
            total_weights += weights.numel()
            zero_weights += int((weights == 0).sum().item())

    sparsity = (zero_weights / total_weights * 100.0) if total_weights > 0 else 0.0
    active_params = total_weights - zero_weights

    # Add biases to total param count
    bias_count = sum(p.numel() for n, p in model.named_parameters() if "bias" in n)
    total_model_params = total_weights + bias_count
    active_total_params = active_params + bias_count

    return {
        "total_weights": total_weights,
        "zero_weights": zero_weights,
        "active_weights": active_params,
        "sparsity_percent": round(sparsity, 2),
        "total_model_parameters": total_model_params,
        "active_model_parameters": active_total_params
    }
