import torch


class FourierConv1d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, size, bias=True, periodic
        =False):
        super(FourierConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not periodic:
            self.size = size
        else:
            self.size = size // 2
        self.weights = torch.nn.Parameter(torch.view_as_real(1 / (
            in_channels * out_channels) * torch.rand(in_channels,
            out_channels, self.size, dtype=torch.cfloat)))
        self.biases = torch.nn.Parameter(torch.view_as_real(1 /
            out_channels * torch.rand(out_channels, self.size, dtype=torch.
            cfloat)))
        self.bias = bias
        self.periodic = periodic

    def forward(self, x):
        if not self.periodic:
            padding = self.size
            x = torch.nn.functional.pad(x, [0, padding])
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.size] = torch.einsum('bix,iox->box', x_ft[:, :,
            :self.size], torch.view_as_complex(self.weights))
        if self.bias:
            out_ft[:, :, :self.size] += torch.view_as_complex(self.biases)
        out = torch.fft.irfft(out_ft, n=x.size(-1))
        if not self.periodic:
            out = out[..., :-padding]
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'size': 4}]
