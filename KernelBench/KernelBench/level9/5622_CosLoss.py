import torch
import torch.nn as nn
import torch.nn.functional as F


class CosLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, state_S, state_T, mask=None):
        """
        This is the loss used in DistilBERT

        :param state_S: Tensor of shape  (batch_size, length, hidden_size)
        :param state_T: Tensor of shape  (batch_size, length, hidden_size)
        :param mask:    Tensor of shape  (batch_size, length)
        """
        if mask is None:
            state_S = state_S.view(-1, state_S.size(-1))
            state_T = state_T.view(-1, state_T.size(-1))
        else:
            mask = mask.to(state_S).unsqueeze(-1).expand_as(state_S)
            state_S = torch.masked_select(state_S, mask).view(-1, mask.size(-1)
                )
            state_T = torch.masked_select(state_T, mask).view(-1, mask.size(-1)
                )
        target = state_S.new(state_S.size(0)).fill_(1)
        loss = F.cosine_embedding_loss(state_S, state_T, target, reduction=
            'mean')
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
