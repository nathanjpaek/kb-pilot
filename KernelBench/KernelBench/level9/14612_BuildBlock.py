import torch
import torch.nn.functional as F
from torch import nn


class BuildBlock(nn.Module):

    def __init__(self, planes=256):
        super(BuildBlock, self).__init__()
        self.planes = planes
        self.toplayer1 = nn.Conv2d(2048, planes, kernel_size=1, stride=1,
            padding=0)
        self.toplayer2 = nn.Conv2d(256, planes, kernel_size=3, stride=1,
            padding=1)
        self.toplayer3 = nn.Conv2d(256, planes, kernel_size=3, stride=1,
            padding=1)
        self.toplayer4 = nn.Conv2d(256, planes, kernel_size=3, stride=1,
            padding=1)
        self.latlayer1 = nn.Conv2d(1024, planes, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(512, planes, kernel_size=1, stride=1,
            padding=0)
        self.latlayer3 = nn.Conv2d(256, planes, kernel_size=1, stride=1,
            padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear', align_corners=True
            ) + y

    def forward(self, c2, c3, c4, c5):
        p5 = self.toplayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.toplayer2(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.toplayer3(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.toplayer4(p2)
        return p2, p3, p4, p5


def get_inputs():
    return [torch.rand([4, 256, 64, 64]), torch.rand([4, 512, 64, 64]),
        torch.rand([4, 1024, 64, 64]), torch.rand([4, 2048, 64, 64])]


def get_init_inputs():
    return [[], {}]
