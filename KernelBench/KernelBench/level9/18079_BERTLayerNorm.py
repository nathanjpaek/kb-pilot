from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class BERTLayerNorm(nn.Module):

    def __init__(self, config, multi_params=None, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        if multi_params is not None:
            self.gamma = nn.Parameter(torch.ones(config.hidden_size_aug))
            self.beta = nn.Parameter(torch.zeros(config.hidden_size_aug))
        else:
            self.gamma = nn.Parameter(torch.ones(config.hidden_size))
            self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
