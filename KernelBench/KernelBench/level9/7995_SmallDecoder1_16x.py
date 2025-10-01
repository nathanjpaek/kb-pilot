import torch
import torch.nn as nn


class SmallDecoder1_16x(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallDecoder1_16x, self).__init__()
        self.fixed = fixed
        self.conv11 = nn.Conv2d(24, 3, 3, 1, 0, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage,
                location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_pwct(self, input):
        out11 = self.conv11(self.pad(input))
        return out11


def get_inputs():
    return [torch.rand([4, 24, 4, 4])]


def get_init_inputs():
    return [[], {}]
