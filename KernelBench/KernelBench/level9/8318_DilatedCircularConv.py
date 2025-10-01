import torch
import torch.nn as nn


class DilatedCircularConv(nn.Module):

    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(DilatedCircularConv, self).__init__()
        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.
            n_adj * 2 + 1, dilation=self.dilation)

    def forward(self, input):
        if self.n_adj != 0:
            input = torch.cat([input[..., -self.n_adj * self.dilation:],
                input, input[..., :self.n_adj * self.dilation]], dim=2)
        return self.fc(input)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4}]
