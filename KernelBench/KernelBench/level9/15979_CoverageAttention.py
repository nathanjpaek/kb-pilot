from _paritybench_helpers import _mock_config
import torch
from torch import nn
import torch.nn.functional as F


class CoverageAttention(nn.Module):

    def __init__(self, config: 'SARGConfig'):
        super(CoverageAttention, self).__init__()
        self.linear_h = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_K = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_cov = nn.Linear(1, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, 1)
        self.register_buffer('masked_bias', torch.tensor(-10000.0))

    def forward(self, h, K, cov, mask=None):
        """
        :param K: (batch_size, src_len, hidden_size)
        :param h: (batch_size, hidden_size)
        :param cov: (batch_size, src_len)
        :param mask:
        :return:
        """
        h_l = self.linear_h(h.unsqueeze(1))
        K_l = self.linear_K(K)
        c_l = self.linear_cov(cov.unsqueeze(2))
        e = self.v(torch.tanh(h_l + K_l + c_l)).squeeze(-1)
        if mask is not None:
            e = e + mask * self.masked_bias
        a = F.softmax(e, dim=-1).unsqueeze(1)
        out = torch.matmul(a, K).squeeze(1)
        a = a.squeeze(1)
        cov = cov + a
        return out, a, cov


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
