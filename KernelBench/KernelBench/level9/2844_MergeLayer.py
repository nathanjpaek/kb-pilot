import torch
import torch.nn as nn


class MergeLayer(nn.Module):

    def __init__(self, h_size, device='cpu'):
        super(MergeLayer, self).__init__()
        self.weight_inbound = nn.Linear(h_size, h_size, bias=True)
        self.weight_outbound = nn.Linear(h_size, h_size, bias=True)
        self.lambda_layer = nn.Linear(h_size * 2, 1, bias=True)
        self.init_params()
        self

    def forward(self, h_inbound, h_outbound):
        h_inbound = self.weight_inbound(h_inbound)
        h_outbound = self.weight_outbound(h_outbound)
        lambda_param = self.lambda_layer(torch.cat([h_inbound, h_outbound],
            dim=1))
        lambda_param = torch.sigmoid(lambda_param)
        h = lambda_param * h_inbound + (1 - lambda_param) * h_outbound
        return h

    def init_params(self):
        nn.init.xavier_normal_(self.weight_inbound.weight, gain=1.414)
        nn.init.xavier_normal_(self.weight_outbound.weight, gain=1.414)
        nn.init.zeros_(self.weight_inbound.bias)
        nn.init.zeros_(self.weight_outbound.bias)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'h_size': 4}]
