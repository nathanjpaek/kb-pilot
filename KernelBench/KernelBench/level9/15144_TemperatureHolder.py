import torch
from torch import nn


class TemperatureHolder(nn.Module):
    """Module that holds a temperature as a learnable value.

    Args:
        initial_log_temperature (float): Initial value of log(temperature).
    """

    def __init__(self, initial_log_temperature=0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(
            initial_log_temperature, dtype=torch.float32))

    def forward(self):
        """Return a temperature as a torch.Tensor."""
        return torch.exp(self.log_temperature)


def get_inputs():
    return []


def get_init_inputs():
    return [[], {}]
