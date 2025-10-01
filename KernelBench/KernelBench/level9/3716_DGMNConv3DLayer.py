from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.init as init


class DGMNConv3DLayer(nn.Module):

    def __init__(self, args):
        self.args = args
        super(DGMNConv3DLayer, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=
            (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3),
            padding=(1, 0, 0))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size
            =(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3),
            padding=(1, 0, 0))
        self.flatten = nn.Flatten()
        self.init_weight()

    def init_weight(self):
        init.xavier_uniform_(self.conv1.weight)
        init.constant_(self.conv1.bias, 0.0)
        init.xavier_uniform_(self.conv2.weight)
        init.constant_(self.conv2.bias, 0.0)

    def forward(self, cube):
        outputs = self.pool1(torch.relu(self.conv1(cube)))
        outputs = self.pool2(torch.relu(self.conv2(outputs)))
        outputs = self.flatten(outputs)
        return outputs


def get_inputs():
    return [torch.rand([4, 1, 64, 64, 64])]


def get_init_inputs():
    return [[], {'args': _mock_config()}]
