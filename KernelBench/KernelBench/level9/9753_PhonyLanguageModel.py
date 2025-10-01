import torch
import torch.nn as nn
import torch.nn.functional as F


class PhonyLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        lm_x = x.clone().detach().float() * 0
        return F.log_softmax(lm_x, 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
