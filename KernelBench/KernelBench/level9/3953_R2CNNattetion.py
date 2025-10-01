import torch
import torch.nn as nn
import torch.utils.data


class R2CNNattetion(nn.Module):

    def __init__(self):
        super(R2CNNattetion, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=4)
        self.deconv2 = nn.ConvTranspose2d(512, 512, kernel_size=(4, 4),
            stride=(2, 2), padding=(1, 1))
        self.deconv3 = nn.ConvTranspose2d(512, 512, kernel_size=(6, 6),
            stride=(4, 4), padding=(1, 1))

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        x2 = self.deconv2(x2)
        x3 = self.deconv3(x3)
        x = x1 + x2 + x3
        return x


def get_inputs():
    return [torch.rand([4, 512, 4, 4])]


def get_init_inputs():
    return [[], {}]
