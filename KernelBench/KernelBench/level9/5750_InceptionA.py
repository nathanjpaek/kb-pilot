import torch
import torch.nn.functional as F


class InceptionA(torch.nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=(1, 1))
        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=(1, 1))
        self.branch5x5 = torch.nn.Conv2d(16, 24, kernel_size=(5, 5), padding=2)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=(3, 3),
            padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=(3, 3),
            padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch1x1(x)
        branch5x5 = self.branch5x5(branch5x5)
        branch3x3 = self.branch1x1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        branch_pool = F.avg_pool2d(x, kernel_size=(3, 3), stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
