import torch
import torch.nn as nn


class MinibatchStdLayer(nn.Module):

    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        group_size = min(self.group_size, x.shape[0])
        s = x.shape
        y = x.view([group_size, -1, s[1], s[2], s[3]])
        y = y.float()
        y = y - torch.mean(y, dim=0, keepdim=True)
        y = torch.mean(y * y, dim=0)
        y = torch.sqrt(y + 1e-08)
        y = torch.mean(torch.mean(torch.mean(y, axis=3, keepdim=True), axis
            =2, keepdim=True), axis=1, keepdim=True)
        y = y.type(x.type())
        y = y.repeat(group_size, 1, s[2], s[3])
        return torch.cat([x, y], axis=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
