import torch
import torch.nn.functional as F


class RefModel2d(torch.nn.Module):
    """The 2D reference model."""

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Conv2d(2, 2, 3, stride=2, bias=False, padding=1,
            padding_mode='reflect')
        self.l2 = torch.nn.BatchNorm2d(2, track_running_stats=False)
        self.l3 = torch.nn.LeakyReLU(0.02)
        self.l4 = torch.nn.Dropout2d(0.5)
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
