import torch
import torch.nn.functional as F


class RefModel1d(torch.nn.Module):
    """The 3D reference model."""

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Conv1d(2, 2, 1, bias=True)
        self.l2 = torch.nn.InstanceNorm1d(2, affine=True)
        self.l3 = torch.nn.ReLU()
        self.l4 = torch.nn.Dropout(0.2)
        self.l5 = torch.nn.AvgPool1d(2)
        self.l7 = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        output = self.l1(x)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        output = F.interpolate(output, mode='linear', scale_factor=2)
        output = self.l7(output)
        return output


def get_inputs():
    return [torch.rand([4, 2, 64])]


def get_init_inputs():
    return [[], {}]
