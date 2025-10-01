import torch
import torch.nn as nn
import torch.quantization.quantize_fx
import torch.utils.data


class KeypointRCNNPredictorNoUpscale(nn.Module):

    def __init__(self, in_channels, num_keypoints):
        super(KeypointRCNNPredictorNoUpscale, self).__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = nn.ConvTranspose2d(input_features,
            num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel //
            2 - 1)
        nn.init.kaiming_normal_(self.kps_score_lowres.weight, mode=
            'fan_out', nonlinearity='relu')
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'num_keypoints': 4}]
