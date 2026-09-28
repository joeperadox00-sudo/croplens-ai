"""
CROPLENS AI: Knowledge Distillation Module
Slide 2 & 7: Knowledge Distillation to transfer dark knowledge from a heavy
teacher vision model into the ultra-lightweight student model.
100% Offline, no external APIs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class DistillationLoss(nn.Module):
    """
    Combined Knowledge Distillation Loss:
    L_total = alpha * L_CE(student_logits, targets) + (1 - alpha) * T^2 * KL_Div(student_soft, teacher_soft)
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.4):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Hard label loss
        loss_ce = self.ce_loss(student_logits, targets)

        # Soft label distillation loss with temperature scaling
        p_student = F.log_softmax(student_logits / self.temperature, dim=1)
        q_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

        loss_kd = self.kl_div(p_student, q_teacher) * (self.temperature ** 2)

        total_loss = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd
        return total_loss, loss_ce, loss_kd


def simulate_distillation_step(
    student_model: nn.Module,
    teacher_logits: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: DistillationLoss
) -> Dict[str, float]:
    """
    Executes a distillation training step.
    """
    student_model.train()
    optimizer.zero_grad()

    student_logits = student_model(features)
    total_loss, loss_ce, loss_kd = criterion(student_logits, teacher_logits, targets)

    total_loss.backward()
    optimizer.step()

    return {
        "total_loss": float(total_loss.item()),
        "ce_loss": float(loss_ce.item()),
        "distill_loss": float(loss_kd.item())
    }
