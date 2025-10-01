import torch
import torch.nn as nn
import torch.nn.functional as F


class AttCeMeanLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, attention_S, attention_T, mask=None):
        """
        Calculate the cross entropy  between attention_S and attention_T, the dim of num_heads is averaged

        :param logits_S: Tensor of shape  (batch_size, num_heads, length, length) or (batch_size, length, length)
        :param logits_T: Tensor of shape  (batch_size, num_heads, length, length) or (batch_size, length, length)
        :param mask:     Tensor of shape  (batch_size, length)
        """
        if len(attention_S.size()) == 4:
            attention_S = attention_S.mean(dim=1)
            attention_T = attention_T.mean(dim=1)
        probs_T = F.softmax(attention_T, dim=-1)
        if mask is None:
            probs_T_select = torch.where(attention_T <= -0.001, torch.
                zeros_like(attention_T), probs_T)
            loss = -(probs_T_select * F.log_softmax(attention_S, dim=-1)).sum(
                dim=-1).mean()
        else:
            mask = mask
            loss = -((probs_T * F.log_softmax(attention_S, dim=-1) * mask.
                unsqueeze(1)).sum(dim=-1) * mask).sum() / mask.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
