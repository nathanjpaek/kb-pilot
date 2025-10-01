import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSample(nn.Sequential):

    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3,
            stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features,
            kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size
            (3)], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.leakyreluA(self.convA(torch.
            cat([up_x, concat_with], dim=1)))))


def get_inputs():
    return [torch.rand([4, 3, 4, 4]), torch.rand([4, 1, 4, 4])]


def get_init_inputs():
    return [[], {'skip_input': 4, 'output_features': 4}]
