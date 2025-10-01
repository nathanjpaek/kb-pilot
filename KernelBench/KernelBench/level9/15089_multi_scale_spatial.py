import torch
import torch.nn as nn


class multi_scale_spatial(nn.Module):

    def __init__(self, limb_blocks):
        super(multi_scale_spatial, self).__init__()
        (self.left_arm, self.right_arm, self.left_leg, self.right_leg, self
            .head_spine) = limb_blocks
        self.maxpool1 = nn.AdaptiveMaxPool2d((1, 20))
        self.maxpool2 = nn.AdaptiveMaxPool2d((1, 20))
        self.maxpool3 = nn.AdaptiveMaxPool2d((1, 20))
        self.maxpool4 = nn.AdaptiveMaxPool2d((1, 20))
        self.maxpool5 = nn.AdaptiveMaxPool2d((1, 20))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 20))

    def forward(self, x):
        ll = self.maxpool1(x[:, :, self.left_leg])
        rl = self.maxpool2(x[:, :, self.right_leg])
        la = self.maxpool3(x[:, :, self.left_arm])
        ra = self.maxpool4(x[:, :, self.right_arm])
        hs = self.maxpool5(x[:, :, self.head_spine])
        multi_sptial = torch.cat((ll, rl, la, ra, hs), dim=-2)
        x = self.avgpool(multi_sptial)
        return x


def get_inputs():
    return [torch.rand([4, 4, 5, 4])]


def get_init_inputs():
    return [[], {'limb_blocks': [4, 4, 4, 4, 4]}]
