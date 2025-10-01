import torch


def _grot90(x, k):
    return torch.rot90(x.roll(k, 2), k, (3, 4))


class ConvP4(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True,
        stride=1, padding=1):
        super().__init__()
        w = torch.empty(out_channels, in_channels, 4, kernel_size, kernel_size)
        self.weight = torch.nn.Parameter(w)
        torch.nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding

    def _rotated(self, w):
        ws = [_grot90(w, k).view(w.size(0), -1, w.size(3), w.size(4)) for k in
            range(4)]
        return torch.cat(ws, 1).view(4 * w.size(0), 4 * w.size(1), w.size(3
            ), w.size(4))

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        w = self._rotated(self.weight)
        y = torch.nn.functional.conv2d(x, w, stride=self.stride, padding=
            self.padding)
        y = y.view(y.size(0), -1, 4, y.size(2), y.size(3))
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
