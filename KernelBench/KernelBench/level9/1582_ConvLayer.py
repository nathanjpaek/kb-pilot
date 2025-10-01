import torch


class ConvLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, kernel_size=1, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.refpadding = torch.nn.ReflectionPad2d(padding)
        self.conv = torch.nn.Conv2d(in_features, out_features, kernel_size,
            stride)

    def forward(self, x):
        x = self.refpadding(x)
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
