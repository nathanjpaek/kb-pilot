import torch
import torch.nn as nn


class self_attn_mini(nn.Module):

    def __init__(self, input_size):
        super(self_attn_mini, self).__init__()
        self.input_size = input_size
        self.key = nn.Linear(input_size, input_size, bias=False)
        self.query = nn.Linear(input_size, input_size, bias=False)
        self.value = nn.Linear(input_size, input_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        q = self.query(input)
        k = self.key(input)
        v = self.value(input)
        wt_mat = q @ torch.transpose(k, 1, 2) / self.input_size
        wt_mat_softmaxed = self.softmax(wt_mat)
        transformed = wt_mat_softmaxed @ v
        return transformed


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
