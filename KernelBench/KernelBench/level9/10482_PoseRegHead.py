import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_fc_layer(in_cn, out_cn):
    x = nn.Linear(in_cn, out_cn)
    x.bias.data.zero_()
    nn.init.normal_(x.weight, 0.0, 0.001)
    return x


class PoseRegHead(nn.Module):

    def __init__(self, dim_in, dim_out, num_units=4096):
        super(PoseRegHead, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out * 4
        self.num_units = num_units
        self.poses_fc1 = _get_fc_layer(self.dim_in, num_units)
        self.poses_fc2 = _get_fc_layer(num_units, num_units)
        self.poses_fc3 = _get_fc_layer(num_units, self.dim_out)

    def forward(self, x, drop_prob=0.0, is_train=False):
        x_flat = x.view(-1, self.dim_in)
        fc1 = self.poses_fc1(x_flat)
        fc1 = F.normalize(fc1, p=2, dim=1)
        fc1 = F.dropout(F.relu(fc1, inplace=True), drop_prob, training=is_train
            )
        fc2 = self.poses_fc2(fc1)
        fc2 = F.normalize(fc2, p=2, dim=1)
        fc2 = F.dropout(F.relu(fc2, inplace=True), drop_prob, training=is_train
            )
        fc3 = self.poses_fc3(fc2)
        fc3 = F.normalize(fc3, p=2, dim=1)
        return F.tanh(fc3)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4}]
