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


class PatchedSelfAttentionLayer(nn.Module):

    def __init__(self, elem_size, embd_size, window_size, use_V=False):
        super(PatchedSelfAttentionLayer, self).__init__()
        self.sa_layer = SelfAttentionLayer(elem_size, embd_size)
        self.window_size = window_size
        self.embd_size = embd_size

    def forward(self, x):
        N, seq_size, elem_size = x.shape
        patches_num = seq_size // self.window_size
        add_patch = False
        if seq_size % self.window_size != 0:
            add_patch = True
        x_patches = x[:, :patches_num * self.window_size, :].reshape(N,
            patches_num, self.window_size, elem_size)
        if add_patch:
            rest_seq_padding = torch.zeros(N, 1, x_patches.shape[2],
                x_patches.shape[3])
            rest_seq_values = x[:, patches_num * self.window_size:, :]
            rest_seq_padding[:, 0, :rest_seq_values.shape[1], :
                ] = rest_seq_values
            x_patches = torch.cat([x_patches, rest_seq_padding], dim=1)
        x_patches, attention_mat = self.sa_layer(x_patches)
        out = x_patches.reshape(x_patches.shape[0], x_patches.shape[1] *
            x_patches.shape[2], x_patches.shape[3])[:, :seq_size, :]
        attention_mat = attention_mat.reshape(attention_mat.shape[0], 
            attention_mat.shape[1] * attention_mat.shape[2], attention_mat.
            shape[3])[:, :seq_size, :self.window_size]
        return out, attention_mat


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'elem_size': 4, 'embd_size': 4, 'window_size': 4}]
