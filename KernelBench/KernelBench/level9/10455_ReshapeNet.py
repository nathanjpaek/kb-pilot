import torch
import torch.nn.functional


class ReshapeNet(torch.nn.Module):

    def __init__(self):
        super(ReshapeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        batch, channels, height, width = x.size()
        x = x * height
        channels = channels + width
        channels = channels - width
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, height, width, channels)
        batch, channels, height, width = x.size()
        height = height + batch
        height = height - batch
        x = torch.transpose(x, 1, 2)
        return x.reshape(-1, channels, height, width)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
