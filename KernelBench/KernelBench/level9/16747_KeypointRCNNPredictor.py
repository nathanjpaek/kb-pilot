import torch
from torch import nn
import torch.utils.data


class KeypointRCNNPredictor(nn.Module):

    def __init__(self, in_channels, num_keypoints):
        super(KeypointRCNNPredictor, self).__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = nn.ConvTranspose2d(input_features,
            num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel //
            2 - 1)
        nn.init.kaiming_normal_(self.kps_score_lowres.weight, mode=
            'fan_out', nonlinearity='relu')
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        return torch.nn.functional.interpolate(x, scale_factor=float(self.
            up_scale), mode='bilinear', align_corners=False,
            recompute_scale_factor=False)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'num_keypoints': 4}]
