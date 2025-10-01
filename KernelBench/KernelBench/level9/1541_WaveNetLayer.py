import torch
import typing as T


class WaveNetLayer(torch.nn.Module):
    """a single gated residual wavenet layer"""

    def __init__(self, channels: 'int', kernel_size: 'int', dilation: 'int'):
        super().__init__()
        self._conv = torch.nn.Conv1d(in_channels=channels, out_channels=
            channels, kernel_size=kernel_size, padding='same', dilation=
            dilation)
        self._conv_skip = torch.nn.Conv1d(in_channels=channels // 2,
            out_channels=channels, kernel_size=1)
        self._conv_out = torch.nn.Conv1d(in_channels=channels // 2,
            out_channels=channels, kernel_size=1)

    def forward(self, x: 'torch.Tensor') ->T.Tuple[torch.Tensor, torch.Tensor]:
        r = x
        x = self._conv(x)
        x, g = x.split(x.size(1) // 2, dim=1)
        x = torch.tanh(x) * torch.sigmoid(g)
        s = self._conv_skip(x)
        x = self._conv_out(x)
        x = (x + r) * torch.tensor(0.5).sqrt()
        return x, s


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'kernel_size': 4, 'dilation': 1}]
