import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, with_logits=True, reduction:
        'str'='mean'):
        """
        https://github.com/mathiaszinnen/focal_loss_torch/blob/main/focal_loss/focal_loss.py
        https://arxiv.org/pdf/1708.02002.pdf
        default gamma from tabel 1(b)
        """
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError('Reduction {} not implemented.'.
                format(reduction))
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma
        self.with_logits = with_logits

    def forward(self, input, target):
        """
        input to be (B, 1) of probabilites (not logits)
        """
        if self.with_logits:
            input = F.sigmoid(input)
        p_t = torch.where(target == 1, input, 1 - input)
        fl = -1 * (1 - p_t) ** self.gamma * torch.log(p_t)
        fl = torch.where(target == 1, fl * self.alpha, fl * (1 - self.alpha))
        return self._reduce(fl)

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
