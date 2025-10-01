import torch
import torch.nn as nn


class Decoder3(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Decoder3, self).__init__()
        self.fixed = fixed
        self.conv31 = nn.Conv2d(256, 128, 3, 1, 0)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv21 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv11 = nn.Conv2d(64, 3, 3, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            self.load_state_dict(torch.load(model, map_location=lambda
                storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.relu(self.conv31(self.pad(input)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y


def get_inputs():
    return [torch.rand([4, 256, 4, 4])]


def get_init_inputs():
    return [[], {}]
