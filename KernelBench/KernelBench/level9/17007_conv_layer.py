import torch
from torch import nn


class conv_layer(nn.Module):
    """Standard convolutional layer. Possible to return pre-activations."""

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
        padding=1, drop=0, batch_norm=False, nl=nn.ReLU(), bias=True, gated
        =False):
        super().__init__()
        if drop > 0:
            self.dropout = nn.Dropout2d(drop)
        self.conv = nn.Conv2d(in_planes, out_planes, stride=stride,
            kernel_size=kernel_size, padding=padding, bias=bias)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_planes)
        if gated:
            self.gate = nn.Conv2d(in_planes, out_planes, stride=stride,
                kernel_size=kernel_size, padding=padding, bias=False)
            self.sigmoid = nn.Sigmoid()
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl == 'none':
            self.nl = nn.ReLU() if nl == 'relu' else nn.LeakyReLU(
                ) if nl == 'leakyrelu' else modules.Identity()

    def forward(self, x, return_pa=False):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.conv(input)) if hasattr(self, 'bn'
            ) else self.conv(input)
        gate = self.sigmoid(self.gate(x)) if hasattr(self, 'gate') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate'
            ) else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl'
            ) else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        return [self.conv]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'out_planes': 4}]
