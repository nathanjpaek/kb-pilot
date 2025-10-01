import torch
import torch.nn as nn
import torch.nn.functional as F


class KdMseLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits_S, logits_T, temperature=1):
        """
        Calculate the mse loss between logits_S and logits_T

        :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
        """
        if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
            temperature = temperature.unsqueeze(-1)
        beta_logits_T = logits_T / temperature
        beta_logits_S = logits_S / temperature
        loss = F.mse_loss(beta_logits_S, beta_logits_T)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
