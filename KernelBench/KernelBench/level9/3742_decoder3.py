import torch
import torch.nn
import torch
import torch.nn as nn


class decoder3(nn.Module):

    def __init__(self, W, v2):
        super(decoder3, self).__init__()
        self.reflecPad7 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(int(256 * W), int(128 * W), 3, 1, 0)
        self.relu7 = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad8 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(int(128 * W), int(128 * W), 3, 1, 0)
        self.relu8 = nn.ReLU(inplace=True)
        self.reflecPad9 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(int(128 * W), int(64 * W), 3, 1, 0)
        self.relu9 = nn.ReLU(inplace=True)
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad10 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(int(64 * W), 32 if v2 else int(64 * W), 3, 1, 0
            )
        self.relu10 = nn.ReLU(inplace=True)
        self.reflecPad11 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(32 if v2 else int(64 * W), 3, 3, 1, 0)

    def forward(self, x):
        out = self.reflecPad7(x)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.unpool(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.unpool2(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        out = out.clamp(0, 1) * 255
        return out


def get_inputs():
    return [torch.rand([4, 1024, 4, 4])]


def get_init_inputs():
    return [[], {'W': 4, 'v2': 4}]
