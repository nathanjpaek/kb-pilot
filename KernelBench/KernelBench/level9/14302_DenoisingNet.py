import torch
import torch.nn as nn


class DenoisingNet(nn.Module):

    def __init__(self, input_vec_size):
        super(DenoisingNet, self).__init__()
        self.linear_layer = nn.Linear(input_vec_size, 1)
        self.elu_layer = nn.ELU()
        self.propensity_net = nn.Sequential(self.linear_layer, self.elu_layer)
        self.list_size = input_vec_size

    def forward(self, input_list):
        output_propensity_list = []
        for i in range(self.list_size):
            click_feature = [torch.unsqueeze(torch.zeros_like(input_list[i]
                ), -1) for _ in range(self.list_size)]
            click_feature[i] = torch.unsqueeze(torch.ones_like(input_list[i
                ]), -1)
            output_propensity_list.append(self.propensity_net(torch.cat(
                click_feature, 1)))
        return torch.cat(output_propensity_list, 1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_vec_size': 4}]
