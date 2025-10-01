import torch


class ConvSample(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=5,
            kernel_size=5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=5, out_channels=5,
            kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = x1 + x2
        return self.relu(x3)


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
