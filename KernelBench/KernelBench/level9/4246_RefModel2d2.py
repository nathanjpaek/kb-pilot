import torch
import torch.nn.functional as F


class RefModel2d2(torch.nn.Module):
    """The 2D reference model."""

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Conv2d(2, 2, 3, padding=1, stride=2,
            padding_mode='circular', bias=False)
        self.l2 = torch.nn.Identity()
        self.l3 = torch.nn.LeakyReLU(0.02)
        self.l4 = torch.nn.Identity()
        self.l5 = torch.nn.AvgPool2d(2, stride=4)
        self.l7 = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        output = self.l1(x)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        output = F.interpolate(output, mode='bilinear', scale_factor=2)
        output = self.l7(output)
        return output


def get_inputs():
    return [torch.rand([4, 2, 4, 4])]


def get_init_inputs():
    return [[], {}]
