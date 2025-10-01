from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(config.hidden_size, 2)

    def forward(self, x, encoder_output):
        y = self.linear(encoder_output)
        return y + x


def get_inputs():
    return [torch.rand([4, 4, 4, 2]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
