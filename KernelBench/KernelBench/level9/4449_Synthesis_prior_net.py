import math
import torch
import torch.nn as nn
import torch.utils.data


class Synthesis_prior_net(nn.Module):
    """
    Decode synthesis prior
    """

    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Synthesis_prior_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5,
            stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1)
            )
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU()
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5,
            stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1)
            )
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU()
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 3,
            stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1)
            )
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.relu3 = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x = self.relu3(self.deconv3(x))
        return x


def get_inputs():
    return [torch.rand([4, 192, 4, 4])]


def get_init_inputs():
    return [[], {}]
