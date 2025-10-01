import torch
import torch.utils.data
import torch.nn.functional as F


class SFU(torch.nn.Module):
    """
    only two input, one input vector and one fusion vector

    Args:
        - input_size:
        - fusions_size:
    Inputs:
        - input: (seq_len, batch, input_size)
        - fusions: (seq_len, batch, fusions_size)
    Outputs:
        - output: (seq_len, batch, input_size)
    """

    def __init__(self, input_size, fusions_size):
        super(SFU, self).__init__()
        self.linear_r = torch.nn.Linear(input_size + fusions_size, input_size)
        self.linear_g = torch.nn.Linear(input_size + fusions_size, input_size)

    def forward(self, input, fusions):
        m = torch.cat((input, fusions), dim=-1)
        r = F.tanh(self.linear_r(m))
        g = F.sigmoid(self.linear_g(m))
        o = g * r + (1 - g) * input
        return o


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'fusions_size': 4}]
