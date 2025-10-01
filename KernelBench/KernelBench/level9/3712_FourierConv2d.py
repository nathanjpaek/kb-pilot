import torch


class FourierConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, size_x, size_y, bias=True,
        periodic=False):
        super(FourierConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not periodic:
            self.size_x = size_x
            self.size_y = size_y
        else:
            self.size_x = size_x // 2
            self.size_y = size_y // 2
        self.weights = torch.nn.Parameter(torch.view_as_real(1 / (
            in_channels * out_channels) * torch.rand(in_channels,
            out_channels, self.size_x, self.size_y, dtype=torch.cfloat)))
        self.biases = torch.nn.Parameter(torch.view_as_real(1 /
            out_channels * torch.rand(out_channels, self.size_x, self.
            size_y, dtype=torch.cfloat)))
        self.bias = bias
        self.periodic = periodic

    def forward(self, x):
        if not self.periodic:
            x = torch.nn.functional.pad(x, [0, self.size_y, 0, self.size_x])
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.size_x, :self.size_y] = torch.einsum(
            'bixy,ioxy->boxy', x_ft[:, :, :self.size_x, :self.size_y],
            torch.view_as_complex(self.weights))
        if self.bias:
            out_ft[:, :, :self.size_x, :self.size_y] += torch.view_as_complex(
                self.biases)
        out = torch.fft.irfft2(out_ft)
        if not self.periodic:
            out = out[..., :self.size_x, :self.size_y]
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'size_x': 4, 'size_y': 4}
        ]
