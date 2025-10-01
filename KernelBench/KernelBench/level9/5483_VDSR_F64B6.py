import torch
import torch.nn as nn


def load_param(model1_path, model2):
    dict_param1 = torch.load(model1_path)
    dict_param2 = dict(model2.named_parameters())
    for name2 in dict_param2:
        if name2 in dict_param1:
            dict_param2[name2].data.copy_(dict_param1[name2].data)
    model2.load_state_dict(dict_param2)
    return model2


class VDSR_F64B6(nn.Module):

    def __init__(self, model=False, fixed=False):
        super(VDSR_F64B6, self).__init__()
        self.fixed = fixed
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv7 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(64, 1, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if model:
            load_param(model, self)
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        y = self.relu(self.conv3(y))
        y = self.relu(self.conv4(y))
        y = self.relu(self.conv5(y))
        y = self.relu(self.conv6(y))
        y = self.relu(self.conv7(y))
        y = self.conv8(y)
        return y

    def forward_stem(self, y):
        y = self.relu(self.conv1(y))
        out1 = y
        y = self.relu(self.conv2(y))
        out2 = y
        y = self.relu(self.conv3(y))
        out3 = y
        y = self.relu(self.conv4(y))
        out4 = y
        y = self.relu(self.conv5(y))
        out5 = y
        y = self.relu(self.conv6(y))
        out6 = y
        y = self.relu(self.conv7(y))
        out7 = y
        y = self.conv8(y)
        out8 = y
        return out1, out2, out3, out4, out5, out6, out7, out8

    def forward_dense(self, y):
        y = self.relu(self.conv1(y))
        out1 = y
        y = self.relu(self.conv2(y))
        out2 = y
        y = self.relu(self.conv3(y))
        out3 = y
        y = self.relu(self.conv4(y))
        out4 = y
        y = self.relu(self.conv5(y))
        out5 = y
        y = self.relu(self.conv6(y))
        out6 = y
        y = self.relu(self.conv7(y))
        out7 = y
        y = self.conv8(y)
        out8 = y
        return out1, out2, out3, out4, out5, out6, out7, out8


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
