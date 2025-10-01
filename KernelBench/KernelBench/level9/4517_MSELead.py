import torch
from torch import nn


class MSELead(nn.Module):

    def __init__(self):
        super(MSELead, self).__init__()
        self.loss_func = nn.MSELoss()

    def forward(self, input, target):
        loss_list = []
        for i in range(input.size(1)):
            loss_list.append(self.loss_func(input[:, i], target[:, i]))
        return torch.mean(torch.stack(loss_list))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
