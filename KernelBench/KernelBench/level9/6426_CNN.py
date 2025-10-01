import torch
from torch.nn import functional as F
from torch import nn


class CNN(nn.Module):
    """Regularization for sparse-data CT and XPCI CT.

    * The CNN has 3 layers:
    inChannels -> Layer 1 -> n_cnn -> Layer 2 ->
    n_cnn -> Layer_3 -> 1 channel

    Args:
        n_cnn (int): Number of output channels in the 1st and 2nd layers.
        imgSize (int): Number of rows/columns in the input image.
        inChannels (int): Number of input channels to the CNN.

    """

    def __init__(self, n_cnn: 'int', imgSize: 'int', inChannels: 'int'):
        super().__init__()
        self.n_cnn = n_cnn
        self.imgSize = imgSize
        self.inChannels = inChannels
        stride = 1
        kernelSize = 3
        pad = (imgSize - (imgSize - kernelSize) / stride - 1) * stride // 2
        pad = int(pad)
        self.conv1 = nn.Conv2d(in_channels=self.inChannels, out_channels=
            self.n_cnn, kernel_size=kernelSize, padding=pad)
        self.conv2 = nn.Conv2d(in_channels=self.n_cnn, out_channels=self.
            n_cnn, kernel_size=kernelSize, padding=pad)
        self.conv3 = nn.Conv2d(in_channels=self.n_cnn, out_channels=1,
            kernel_size=kernelSize, padding=pad)

    def forward(self, x_concat):
        x = F.relu(self.conv1(x_concat))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_cnn': 4, 'imgSize': 4, 'inChannels': 4}]
