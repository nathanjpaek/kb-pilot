import torch
import torch.nn as nn


class MockOpticalFlowModel(nn.Module):

    def __init__(self, img_channels):
        super().__init__()
        self.model = nn.Conv2d(img_channels * 2, 2, kernel_size=1)

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], dim=-3)
        return self.model(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'img_channels': 4}]
