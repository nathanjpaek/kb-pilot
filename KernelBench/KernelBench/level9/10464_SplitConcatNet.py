import torch
import torch.nn.functional


class SplitConcatNet(torch.nn.Module):

    def __init__(self):
        super(SplitConcatNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(1, 3, kernel_size=1, stride=1)
        self.conv3 = torch.nn.Conv2d(1, 3, kernel_size=1, stride=1)

    def forward(self, x, y):
        x = self.conv1(x)
        y = self.conv1(y)
        x1, x2, x3 = torch.split(x, split_size_or_sections=1, dim=1)
        _y1, y2, y3 = torch.split(y, split_size_or_sections=1, dim=1)
        x4 = (x3 - x1) * x2
        xy1 = torch.concat([x2, y2], 1)
        xy2 = torch.concat([x1, y3], 1)
        return self.conv3(x3), self.conv2(x2), x4, xy1 - xy2


def get_inputs():
    return [torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
