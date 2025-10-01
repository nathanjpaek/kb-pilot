import torch
import torch.nn as nn


class Spatial_Attention(nn.Module):

    def __init__(self, input_dim):
        super(Spatial_Attention, self).__init__()
        self.att_conv1 = nn.Conv2d(input_dim, 1, kernel_size=(1, 1),
            padding=0, stride=1, bias=False)
        self.att_act2 = nn.Softplus(beta=1, threshold=20)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        att_score = self.att_act2(self.att_conv1(x))
        return att_score


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
