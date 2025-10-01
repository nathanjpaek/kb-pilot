import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
  This class defines the encoder architecture
  """

    def __init__(self, input_size, hidden_size, bottleneck):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.mean = nn.Linear(hidden_size, bottleneck)
        self.var = nn.Linear(hidden_size, bottleneck)
        nn.init.normal_(self.linear1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.mean.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.var.weight, mean=0.0, std=0.01)

    def forward(self, x):
        mean = self.mean(torch.tanh(self.linear1(x)))
        log_var = self.var(torch.tanh(self.linear1(x)))
        return mean, log_var


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'bottleneck': 4}]
