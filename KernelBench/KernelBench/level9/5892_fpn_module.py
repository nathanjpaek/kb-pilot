import torch
import torch.nn as nn
import torch.nn.functional as F


class fpn_module(nn.Module):

    def __init__(self, numClass):
        super(fpn_module, self).__init__()
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0
            )
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.smooth1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1
            )
        self.smooth2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1
            )
        self.smooth3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1
            )
        self.smooth4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1
            )
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0
            )
        self.classify = nn.Conv2d(128 * 4, numClass, kernel_size=3, stride=
            1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), mode='bilinear')
        p4 = F.interpolate(p4, size=(H, W), mode='bilinear')
        p3 = F.interpolate(p3, size=(H, W), mode='bilinear')
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, c2, c3, c4, c5):
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p5 = self.smooth1_2(self.smooth1_1(p5))
        p4 = self.smooth2_2(self.smooth2_1(p4))
        p3 = self.smooth3_2(self.smooth3_1(p3))
        p2 = self.smooth4_2(self.smooth4_1(p2))
        output = self.classify(self._concatenate(p5, p4, p3, p2))
        return output


def get_inputs():
    return [torch.rand([4, 256, 64, 64]), torch.rand([4, 512, 64, 64]),
        torch.rand([4, 1024, 64, 64]), torch.rand([4, 2048, 64, 64])]


def get_init_inputs():
    return [[], {'numClass': 4}]
