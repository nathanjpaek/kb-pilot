import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_PART(nn.Module):

    def __init__(self, filter_channels, merge_layer=0, res_layers=[], norm=
        'group', num_parts=2, last_op=None):
        super(MLP_PART, self).__init__()
        self.num_parts = num_parts
        self.fc_parts_0 = nn.Conv1d(filter_channels[0], 512, 1)
        self.fc_parts_1 = nn.Conv1d(512, 256, 1)
        self.fc_parts_out = nn.Conv1d(256, num_parts, 1)
        self.fc_parts_softmax = nn.Softmax(1)
        self.part_0 = nn.Conv1d(filter_channels[0], 256 * num_parts, 1)
        self.part_1 = nn.Conv1d(256 * num_parts, 128 * num_parts, 1, groups
            =num_parts)
        self.part_2 = nn.Conv1d(128 * num_parts, 128 * num_parts, 1, groups
            =num_parts)
        self.part_out = nn.Conv1d(128 * num_parts, num_parts, 1, groups=
            num_parts)
        self.actvn = nn.ReLU()
        self.last = last_op

    def forward(self, feature):
        """
        feature may include multiple view inputs
        args:
            feature: [B, C_in, N]
        return:
            [B, C_out, N] occupancy prediction
            [B, num_parts, N] parts prediction
        """
        net_parts = self.actvn(self.fc_parts_0(feature))
        net_parts = F.relu(self.fc_parts_1(net_parts))
        out_parts = self.fc_parts_out(net_parts)
        parts_softmax = self.fc_parts_softmax(out_parts)
        net_full = self.actvn(self.part_0(feature))
        net_full = self.actvn(self.part_1(net_full))
        net_full = self.actvn(self.part_2(net_full))
        net_full = self.part_out(net_full)
        net_full *= parts_softmax
        out_full = net_full.mean(1).view(net_full.shape[0], 1, -1)
        if self.last:
            out_full = self.last(out_full)
        return out_full, out_parts


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'filter_channels': [4, 4]}]
