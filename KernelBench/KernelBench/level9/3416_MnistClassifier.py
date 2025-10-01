from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class MnistClassifier(nn.Module):

    def __init__(self, config):
        super(MnistClassifier, self).__init__()
        self.config = config
        self.h = self.config['image_h']
        self.w = self.config['image_w']
        self.out_dim = self.config['class_num']
        self.fc1 = nn.Linear(self.h * self.w, 16)
        self.fc2 = nn.Linear(16, self.out_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x_hidd = x
        x = x ** 2
        x = self.fc2(x)
        return x, x_hidd


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(image_h=4, image_w=4, class_num=4)}]
