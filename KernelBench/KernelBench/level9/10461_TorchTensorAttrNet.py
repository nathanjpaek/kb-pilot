import torch
import torch.nn.functional


class TorchTensorAttrNet(torch.nn.Module):

    def __init__(self):
        super(TorchTensorAttrNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = x * x.size(1)
        return x.view(1, -1)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
