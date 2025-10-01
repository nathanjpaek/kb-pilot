import logging
import torch


class OneConv3d(torch.nn.Module):
    """OneConv3d.
    """

    def __init__(self, out_channels=2):
        super().__init__()
        self.layer = torch.nn.Conv3d(in_channels=1, out_channels=
            out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        logging.debug(f'x.shape={x.shape!r}')
        out = self.layer(x)
        logging.debug(f'out.shape={out.shape!r}')
        return out


def get_inputs():
    return [torch.rand([4, 1, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
