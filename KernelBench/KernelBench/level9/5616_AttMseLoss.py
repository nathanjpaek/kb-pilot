import torch
import torch.nn as nn
import torch.nn.functional as F


class AttMseLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, attention_S, attention_T, mask=None):
        """
        Calculate the mse loss between attention_S and attention_T.

        :param logits_S: Tensor of shape  (batch_size, num_heads, length, length)
        :param logits_T: Tensor of shape  (batch_size, num_heads, length, length)
        :param mask:     Tensor of shape  (batch_size, length)
        """
        if mask is None:
            attention_S_select = torch.where(attention_S <= -0.001, torch.
                zeros_like(attention_S), attention_S)
            attention_T_select = torch.where(attention_T <= -0.001, torch.
                zeros_like(attention_T), attention_T)
            loss = F.mse_loss(attention_S_select, attention_T_select)
        else:
            mask = mask.unsqueeze(1).expand(-1, attention_S.size(1), -1)
            valid_count = torch.pow(mask.sum(dim=2), 2).sum()
            loss = (F.mse_loss(attention_S, attention_T, reduction='none') *
                mask.unsqueeze(-1) * mask.unsqueeze(2)).sum() / valid_count
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
