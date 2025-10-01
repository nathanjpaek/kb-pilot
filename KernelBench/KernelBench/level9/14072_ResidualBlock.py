import torch
from torch import nn
from torch.nn import Linear
from math import sqrt
from torch.nn import Conv1d
import torch.utils.data
import torch.optim
import torch.distributions


class ResidualBlock(nn.Module):

    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels,
            3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(residual_channels, residual_channels
            )
        self.conditioner_projection = Conv1d(encoder_hidden, 2 *
            residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 *
            residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1
            )
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'encoder_hidden': 4, 'residual_channels': 4, 'dilation': 1}]
