import torch
import torch.nn as nn
import torch.utils.data


class visual_context(nn.Module):

    def __init__(self):
        super(visual_context, self).__init__()
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, visual_feature):
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 
            1, 2))
        visual_feature = visual_feature.squeeze(3)
        return visual_feature


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
