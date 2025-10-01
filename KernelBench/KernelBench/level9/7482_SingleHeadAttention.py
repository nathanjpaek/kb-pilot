from _paritybench_helpers import _mock_config
import math
import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):

    def __init__(self, cfg):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = cfg.embedding_dim
        self.embedding_dim = cfg.embedding_dim
        self.value_dim = self.embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = cfg.tanh_clipping
        self.norm_factor = 1 / math.sqrt(self.key_dim)
        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q
        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim
        h_flat = h.reshape(-1, input_dim)
        q_flat = q.reshape(-1, input_dim)
        shape_k = batch_size, target_size, -1
        shape_q = batch_size, n_query, -1
        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(h_flat, self.w_key).view(shape_k)
        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        U = self.tanh_clipping * torch.tanh(U)
        if mask is not None:
            mask = mask.view(batch_size, 1, target_size).expand_as(U)
            U[mask.bool()] = -100000000.0
        attention = torch.log_softmax(U, dim=-1)
        out = attention
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'cfg': _mock_config(embedding_dim=4, tanh_clipping=4)}]
