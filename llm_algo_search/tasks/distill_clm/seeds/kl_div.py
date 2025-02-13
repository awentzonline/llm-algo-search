"""Train on KL divergence"""
from torch import nn
import torch.nn.functional as F


class API(nn.Module):
    def forward(self, student_logits, target_logits, labels):
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(target_logits, dim=-1),
            reduction='batchmean'
        )
        return loss
