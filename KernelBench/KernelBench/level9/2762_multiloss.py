import torch
import torch.nn as nn


class multiloss(nn.Module):

    def __init__(self, objective_num):
        super(multiloss, self).__init__()
        self.objective_num = objective_num
        self.log_var = nn.Parameter(torch.zeros(self.objective_num))

    def forward(self, losses):
        for i in range(len(losses)):
            precision = torch.exp(-self.log_var[i])
            if i == 0:
                loss = precision * losses[i] + self.log_var[i]
            else:
                loss += precision * losses[i] + self.log_var[i]
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'objective_num': 4}]
