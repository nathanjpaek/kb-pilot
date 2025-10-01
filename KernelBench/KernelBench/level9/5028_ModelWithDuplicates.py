import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import *
import torch.optim.lr_scheduler
import torch.onnx
import torch.testing


class ModelWithDuplicates(nn.Module):

    def __init__(self):
        super(ModelWithDuplicates, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.post_conv1 = nn.ModuleList([nn.ReLU(), nn.Tanh()])
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.post_conv2 = self.post_conv1
        self.expected_mlist_to_dmlist = OrderedDict([('post_conv1', [
            'post_conv1']), ('post_conv2', ['post_conv2'])])
        self.expected_list_contents_name_changes = OrderedDict([(
            'post_conv1.0', 'post_conv1_0'), ('post_conv1.1',
            'post_conv1_1'), ('post_conv2.0', 'post_conv2_0'), (
            'post_conv2.1', 'post_conv2_1')])

    def forward(self, x):
        x = self.conv1(x)
        for m in self.post_conv1:
            x = m(x)
        x = self.conv2(x)
        for m in self.post_conv2:
            x = m(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
