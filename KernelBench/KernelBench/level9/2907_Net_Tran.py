import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class Net_Tran(nn.Module):

    def __init__(self):
        super(Net_Tran, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 1, output_padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 8, 3, 2, 1, output_padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.deconv1(out)
        out = self.conv2(out)
        out = self.deconv2(out)
        return out


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
