"""Train on labels"""
from torch import nn
import torch.nn.functional as F


class API(nn.Module):
    def forward(self, student_logits, target_logits, labels):
        student_logits = student_logits.view(-1, student_logits.shape[-1])
        labels = labels.reshape(-1)
        loss = F.cross_entropy(student_logits, labels)
        return loss
