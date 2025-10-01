import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, left_channel, right_channel, out_channel):
        super(MLP, self).__init__()
        self.left = nn.Linear(left_channel, 128)
        self.right = nn.Linear(right_channel, 128)
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, out_channel)

    def forward(self, left, right):
        left_res = self.left(left)
        right_res = self.right(right)
        tmp = torch.cat([left_res, right_res], dim=1)
        tmp = torch.relu(tmp)
        tmp = torch.relu(self.l1(tmp))
        tmp = self.l2(tmp)
        return tmp


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'left_channel': 4, 'right_channel': 4, 'out_channel': 4}]
