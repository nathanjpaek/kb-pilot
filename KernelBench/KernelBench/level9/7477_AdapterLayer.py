from _paritybench_helpers import _mock_config
import math
import torch
import torch.nn as nn


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class AdapterLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.adapter_linear1 = nn.Linear(config.hidden_size, config.
            adapter_size)
        self.gelu = gelu
        self.adapter_linear2 = nn.Linear(config.adapter_size, config.
            hidden_size)

    def forward(self, input_tensor):
        net = self.adapter_linear1(input_tensor)
        net = self.gelu(net)
        net = self.adapter_linear2(net)
        return net + input_tensor


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, adapter_size=4)}]
