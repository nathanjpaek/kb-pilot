import torch
import torch.serialization
import torch
import torch.utils.data


class ResBlock(torch.nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=64, out_channels=64,
            kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64,
            kernel_size=3, stride=1, padding=1)

    def forward(self, frames):
        """
        Args:
            frames: 1x64xHxW

        Returns: 1x64xHxW

        """
        res = self.conv1(frames)
        res = torch.nn.functional.relu(res)
        res = self.conv2(res)
        return frames + res


class Feature(torch.nn.Module):

    def __init__(self):
        super(Feature, self).__init__()
        self.preconv = torch.nn.Conv2d(in_channels=3, out_channels=64,
            kernel_size=3, stride=1, padding=1)
        self.resblock_1 = ResBlock()
        self.resblock_2 = ResBlock()
        self.resblock_3 = ResBlock()
        self.resblock_4 = ResBlock()
        self.resblock_5 = ResBlock()
        self.conv1x1 = torch.nn.Conv2d(in_channels=64, out_channels=3,
            kernel_size=1)

    def forward(self, frame):
        """
        Args:
            frame: 1x3xHxW

        Returns: 1x3xHxW

        """
        x = self.preconv(frame)
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        x = self.resblock_4(x)
        x = self.resblock_5(x)
        x = self.conv1x1(x)
        return frame - x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
