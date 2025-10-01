import torch
import torch.nn.functional as F


class SimpleCNN(torch.nn.Module):

    def __init__(self, in_ch=1, out_ch=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1,
            padding=1)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=
            1, padding=1)
        self.conv3 = torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=
            1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
