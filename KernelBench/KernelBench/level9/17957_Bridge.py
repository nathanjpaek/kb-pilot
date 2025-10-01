import torch
import torch.nn as nn
import torch.nn.functional as F


class Bridge(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(Bridge, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            padding=1)
        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, encoder_feature_map, decoder_feature_map):
        upsampled_decoder_map = self.bilinear_upsampling(decoder_feature_map,
            encoder_feature_map.shape)
        concatenated_maps = torch.cat((upsampled_decoder_map,
            encoder_feature_map), dim=1)
        return self.act2(self.conv2(self.act1(self.conv1(concatenated_maps))))

    def bilinear_upsampling(self, x, shape):
        return F.interpolate(x, size=(shape[2], shape[3]), mode='bilinear',
            align_corners=True)


def get_inputs():
    return [torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
