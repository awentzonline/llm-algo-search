import torch
from torch import nn
import numpy as np


class QuantilesOutput(nn.Module):
    """Non-crossing quantiles"""

    def __init__(self, model_dims, quantiles=None, num_quantiles=7):
        super().__init__()

        if quantiles is None:
            quantiles = torch.linspace(0, 1, num_quantiles)
        elif not torch.is_tensor(quantiles):
            quantiles = torch.FloatTensor(quantiles)
        self.register_buffer('quantiles', quantiles[None, ...])

        self.predict_quantile_logits = nn.Sequential(
            nn.Linear(model_dims, len(quantiles)),
            nn.Softmax(-1),
        )
        self.predict_scale = nn.Sequential(
            nn.Linear(model_dims, 1),
            nn.Softplus(),
        )
        self.predict_offset = nn.Sequential(
            nn.Linear(model_dims, 1),
        )

    def forward(self, x):
        logits = self.predict_quantile_logits(x)
        scale = self.predict_scale(x)
        offset = self.predict_offset(x)
        quantiles = torch.cumsum(logits, -1)
        output = quantiles * scale + offset
        return output


def quantile_loss(y_pred, y_true, quantiles):
    """
    Computes the quantile loss for multiple quantiles.

    Parameters:
    y_pred (torch.Tensor): Predicted values, shape (batch_size, num_quantiles)
    y_true (torch.Tensor): True values, shape (batch_size, 1)
    quantiles (torch.Tensor): Tensor of quantiles to use

    Returns:
    torch.Tensor: The computed quantile loss
    """
    assert y_pred.shape[1] == len(quantiles), "Mismatch between predictions and quantiles"

    if len(y_true.shape) == 1:
        y_true = y_true.unsqueeze(1)  # Ensure shape compatibility
    errors = y_true - y_pred
    if torch.is_tensor(quantiles):
        quantiles = quantiles.to(y_pred.device)
    else:
        quantiles = torch.tensor(quantiles, device=y_pred.device).unsqueeze(0)  # Shape (1, num_quantiles)

    loss = torch.maximum(quantiles * errors, (quantiles - 1) * errors)
    return loss.mean()


def quantile_huber_loss(y_pred, y_true, quantiles, delta=1.0):
    """
    Computes the quantile Huber loss for multiple quantiles.

    Parameters:
    y_pred (torch.Tensor): Predicted values, shape (batch_size, num_quantiles)
    y_true (torch.Tensor): True values, shape (batch_size, 1)
    quantiles (torch.Tensor): Tensor of quantiles to use
    delta (float): Huber loss threshold

    Returns:
    torch.Tensor: The computed quantile Huber loss
    """
    assert y_pred.shape[1] == len(quantiles), "Mismatch between predictions and quantiles"

    if len(y_true.shape) == 1:
        y_true = y_true.unsqueeze(1)  # Ensure shape compatibility
    if torch.is_tensor(quantiles):
        quantiles = quantiles.to(y_pred.device)
    else:
        quantiles = torch.tensor(quantiles, device=y_pred.device).unsqueeze(0)  # Shape (1, num_quantiles)
    errors = y_true - y_pred
    abs_errors = torch.abs(errors)
    squared_loss = 0.5 * errors ** 2
    linear_loss = delta * (abs_errors - 0.5 * delta)
    huber_loss = torch.where(abs_errors < delta, squared_loss, linear_loss)

    loss = torch.abs((quantiles - (errors < 0).float()) * huber_loss).sum(-1)

    return loss.mean()
