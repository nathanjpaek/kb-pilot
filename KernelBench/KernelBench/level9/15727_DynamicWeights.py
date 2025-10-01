import torch
import torch.utils.data
from torch import nn


class DynamicWeights(nn.Module):

    def __init__(self, channels):
        super(DynamicWeights, self).__init__()
        self.cata = nn.Conv2d(channels, 9, 3, padding=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.unfold1 = nn.Unfold(kernel_size=(3, 3), padding=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        blur_depth = x
        dynamic_filter1 = self.cata(blur_depth)
        N, _C, H, W = blur_depth.size()
        dynamic_filter1 = self.softmax(dynamic_filter1.permute(0, 2, 3, 1).
            contiguous().view(N * H * W, -1))
        xd_unfold1 = self.unfold1(blur_depth)
        xd_unfold1 = xd_unfold1.contiguous().view(N, blur_depth.size()[1], 
            9, H * W).permute(0, 3, 1, 2).contiguous().view(N * H * W,
            blur_depth.size()[1], 9)
        out1 = torch.bmm(xd_unfold1, dynamic_filter1.unsqueeze(2))
        out1 = out1.view(N, H, W, blur_depth.size()[1]).permute(0, 3, 1, 2
            ).contiguous()
        out = self.gamma * out1 + x
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
