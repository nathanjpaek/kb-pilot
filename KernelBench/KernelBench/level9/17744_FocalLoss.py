import torch
from torchvision.transforms import functional as F
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma: 'int'=2) ->None:
        super().__init__()
        self.gamma = gamma

    def forward(self, output: 'torch.Tensor', target: 'torch.Tensor'
        ) ->torch.Tensor:
        max_val = (-output).clamp(min=0)
        loss = output - output * target + max_val + ((-max_val).exp() + (-
            output - max_val).exp()).log()
        invprobs = F.logsigmoid(-output * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
