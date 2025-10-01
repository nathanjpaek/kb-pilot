import torch
import torch.nn as nn
import torch.nn.functional as F


class AttCeLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, attention_S, attention_T, mask=None):
        """
        Calculate the cross entropy  between attention_S and attention_T.

        :param logits_S: Tensor of shape  (batch_size, num_heads, length, length)
        :param logits_T: Tensor of shape  (batch_size, num_heads, length, length)
        :param mask:     Tensor of shape  (batch_size, length)
        """
        probs_T = F.softmax(attention_T, dim=-1)
        if mask is None:
            probs_T_select = torch.where(attention_T <= -0.001, torch.
                zeros_like(attention_T), probs_T)
            loss = -(probs_T_select * F.log_softmax(attention_S, dim=-1)).sum(
                dim=-1).mean()
        else:
            mask = mask.unsqueeze(1).expand(-1, attention_S.size(1), -1)
            loss = -((probs_T * F.log_softmax(attention_S, dim=-1) * mask.
                unsqueeze(2)).sum(dim=-1) * mask).sum() / mask.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
