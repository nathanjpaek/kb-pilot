import torch
import torch.nn as nn
import torch.multiprocessing


class SE_layer_3d(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):
        super(SE_layer_3d, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.globalAvgPool = nn.AdaptiveAvgPool3d(1)

    def forward(self, input_tensor):
        b, c, _d, _w, _h = input_tensor.size()
        squeeze_tensor = self.globalAvgPool(input_tensor).view(b, c).float()
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        output_tensor = torch.mul(input_tensor, fc_out_2.view(b, c, 1, 1, 1))
        return output_tensor


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
