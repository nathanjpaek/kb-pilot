import torch
import torch.nn as nn


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes,
        batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.h_init = nn.Parameter(torch.zeros(num_hidden, 1),
            requires_grad=False)
        self.w_hx = nn.Parameter(nn.init.orthogonal_(torch.Tensor(
            num_hidden, input_dim).normal_(mean=0, std=0.0001)))
        self.w_hh = nn.Parameter(nn.init.orthogonal_(torch.Tensor(
            num_hidden, num_hidden).normal_(mean=0, std=0.0001)))
        self.b_h = nn.Parameter(torch.Tensor(num_hidden, 1).zero_())
        self.w_ph = nn.Parameter(torch.Tensor(num_classes, num_hidden).
            normal_(mean=0, std=0.0001))
        self.b_p = nn.Parameter(torch.Tensor(num_classes, 1).zero_())

    def forward(self, x):
        h_t = self.h_init
        tanh = nn.Tanh()
        for step in range(self.seq_length):
            h_t = tanh(self.w_hx @ x[:, step].t() + self.w_hh @ h_t + self.b_h)
        p_t = self.w_ph @ h_t + self.b_p
        return p_t.t()


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'seq_length': 4, 'input_dim': 4, 'num_hidden': 4,
        'num_classes': 4, 'batch_size': 4}]
