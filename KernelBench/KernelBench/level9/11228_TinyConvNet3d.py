import torch


class TinyConvNet3d(torch.nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(in_channels, 16, 1)
        self.nlin1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv3d(16, 64, 1)
        self.nlin2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv3d(64, out_channels, 1)
        self.nlin3 = torch.nn.Sigmoid()

    def forward(self, x):
        return torch.nn.Sequential(self.conv1, self.nlin1, self.conv2, self
            .nlin2, self.conv3, self.nlin3)(x)


def get_inputs():
    return [torch.rand([4, 1, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
