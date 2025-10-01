from _paritybench_helpers import _mock_config
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch as th
from torchvision.ops.boxes import *
from torchvision.transforms.functional import *


class Grounding(nn.Module):

    def __init__(self, cfgT, cfgI, heads=1):
        super(Grounding, self).__init__()
        projection = cfgI.hidden_size // 2
        self.num_attention_heads = heads
        self.attention_head_size = int(projection // self.num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.Q = nn.Linear(cfgT.hidden_size, self.all_head_size)
        self.K = nn.Linear(cfgI.hidden_size, self.all_head_size)
        self.cfgT = cfgT
        self.cfgI = cfgI

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, encT, encI, mask):
        Q = self.Q(encT)
        K = self.K(encI)
        Q = self.transpose(Q)
        K = self.transpose(K)
        logits = th.matmul(Q, K.transpose(-1, -2))
        logits = logits / math.sqrt(self.attention_head_size)
        logits = logits + mask
        return logits.squeeze()


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'cfgT': _mock_config(hidden_size=4), 'cfgI': _mock_config(
        hidden_size=4)}]
