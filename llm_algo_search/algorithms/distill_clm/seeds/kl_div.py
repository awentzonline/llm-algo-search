"""Baseline distillation loss"""
from torch import nn
import torch.nn.functional as F


class API(nn.Module):
    def forward(self, student_logits, target_logits):
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.log_softmax(target_logits, dim=-1), log_target=True,
            reduction='batchmean'
        )
        return loss
