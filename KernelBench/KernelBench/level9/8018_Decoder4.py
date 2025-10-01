import torch
import torch.nn as nn


class Decoder4(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Decoder4, self).__init__()
        self.fixed = fixed
        self.conv41 = nn.Conv2d(512, 256, 3, 1, 0)
        self.conv34 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 0)
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
        y = self.relu(self.conv41(self.pad(input)))
        y = self.unpool(y)
        y = self.relu(self.conv34(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv31(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y


def get_inputs():
    return [torch.rand([4, 512, 4, 4])]


def get_init_inputs():
    return [[], {}]
