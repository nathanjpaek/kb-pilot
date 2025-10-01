import torch
import torch.nn as nn
import torch.utils.data


class GroupLinear(nn.Module):

    def __init__(self, in_features, out_features, groups, bias=True):
        super(GroupLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(groups, out_features //
            groups, in_features // groups))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(groups, out_features //
                groups))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        """
        Args:
            input (Tensor): shape (*, in_features)
        """
        batch_size = input.shape[:-1]
        if self.bias is not None:
            output = self.weight @ input.reshape(*batch_size, self.groups, 
                self.in_features // self.groups, 1) + self.bias[..., None]
        else:
            output = self.weight @ input.reshape(*batch_size, self.groups, 
                self.in_features // self.groups, 1)
        return output.reshape(*batch_size, self.out_features)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'groups': 1}]
