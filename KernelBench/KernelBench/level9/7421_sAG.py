import torch
import torch.nn as nn


class sAG(nn.Module):

    def __init__(self, num_channels_in_enc, num_channels_in_dec):
        super(sAG, self).__init__()
        self.num_channels_in_enc = num_channels_in_enc
        self.num_channels_in_dec = num_channels_in_dec
        self.ch_max_pool_enc = nn.MaxPool3d(kernel_size=(self.
            num_channels_in_enc, 1, 1))
        self.ch_avg_pool_enc = nn.AvgPool3d(kernel_size=(self.
            num_channels_in_enc, 1, 1))
        self.conv1_enc = nn.Conv2d(in_channels=self.num_channels_in_enc,
            out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv2_enc = nn.Conv2d(in_channels=3, out_channels=1,
            kernel_size=7, stride=1, padding=3)
        self.ch_max_pool_dec = nn.MaxPool3d(kernel_size=(self.
            num_channels_in_dec, 1, 1))
        self.ch_avg_pool_dec = nn.AvgPool3d(kernel_size=(self.
            num_channels_in_dec, 1, 1))
        self.conv1_dec = nn.Conv2d(in_channels=self.num_channels_in_dec,
            out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv2_dec = nn.Conv2d(in_channels=3, out_channels=1,
            kernel_size=7, stride=1, padding=3)

    def forward(self, enc, dec):
        enc = torch.cat(tensors=(self.ch_max_pool_enc(enc), self.
            ch_avg_pool_enc(enc), self.conv1_enc(enc)), dim=1)
        enc = self.conv2_enc(enc)
        dec = torch.cat(tensors=(self.ch_max_pool_dec(dec), self.
            ch_avg_pool_dec(dec), self.conv1_dec(dec)), dim=1)
        dec = self.conv2_dec(dec)
        out = torch.sigmoid(enc + dec)
        return out

    def initialize_weights(self):
        for layer in [self.conv1_enc, self.conv1_dec, self.conv2_enc, self.
            conv2_dec]:
            nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)
            nn.init.constant_(layer.bias.data, 0.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels_in_enc': 4, 'num_channels_in_dec': 4}]
