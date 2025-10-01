import torch
import numpy as np
import torch.nn as nn


class Encoder5(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Encoder5, self).__init__()
        self.fixed = fixed
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv0.weight = nn.Parameter(torch.from_numpy(np.array([[[[0]],
            [[0]], [[255]]], [[[0]], [[255]], [[0]]], [[[255]], [[0]], [[0]
            ]]])).float())
        self.conv0.bias = nn.Parameter(torch.from_numpy(np.array([-103.939,
            -116.779, -123.68])).float())
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv34 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 0)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv44 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = load_lua(model)
                load_param(t7_model, 0, self.conv0)
                load_param(t7_model, 2, self.conv11)
                load_param(t7_model, 5, self.conv12)
                load_param(t7_model, 9, self.conv21)
                load_param(t7_model, 12, self.conv22)
                load_param(t7_model, 16, self.conv31)
                load_param(t7_model, 19, self.conv32)
                load_param(t7_model, 22, self.conv33)
                load_param(t7_model, 25, self.conv34)
                load_param(t7_model, 29, self.conv41)
                load_param(t7_model, 32, self.conv42)
                load_param(t7_model, 35, self.conv43)
                load_param(t7_model, 38, self.conv44)
                load_param(t7_model, 42, self.conv51)
            else:
                self.load_state_dict(torch.load(model, map_location=lambda
                    storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.conv0(input)
        y = self.relu(self.conv11(self.pad(y)))
        y = self.relu(self.conv12(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv21(self.pad(y)))
        y = self.relu(self.conv22(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv31(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv34(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv41(self.pad(y)))
        y = self.relu(self.conv42(self.pad(y)))
        y = self.relu(self.conv43(self.pad(y)))
        y = self.relu(self.conv44(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv51(self.pad(y)))
        return y


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
