import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=
            num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced,
            out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))
        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))
        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size,
            num_channels, 1, 1, W), squeeze_tensor_h.view(batch_size,
            num_channels, 1, H, 1), squeeze_tensor_d.view(batch_size,
            num_channels, D, 1, 1)])
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.
            conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)
        return output_tensor


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
