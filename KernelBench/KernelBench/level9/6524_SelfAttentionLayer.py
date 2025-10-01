import math
import torch
import torch.utils.data
import torch
from torch import nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):

    def __init__(self, elem_size, embd_size):
        super(SelfAttentionLayer, self).__init__()
        self.embd_size = embd_size
        self.query_lin = nn.Linear(elem_size, embd_size)
        self.key_lin = nn.Linear(elem_size, embd_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        _N, _num_patches, _seq_size, elem_size = x.shape
        Q = F.relu(self.query_lin(x))
        K = F.relu(self.key_lin(x))
        attention_mat = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(
            elem_size)
        attention_mat = F.softmax(attention_mat, dim=-1)
        new_values = torch.matmul(attention_mat, x)
        out = new_values
        out = x + out
        return out, attention_mat


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'elem_size': 4, 'embd_size': 4}]
