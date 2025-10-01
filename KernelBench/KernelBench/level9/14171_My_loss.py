import torch
import torch.utils.data
import torch._utils
import torch.nn.parallel
import torch.optim
from torch.autograd import Variable as Variable


class My_loss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        cccs = 0
        for i in range(x.size(-1)):
            x_i = x[:, i]
            y_i = y[:, i]
            if len(x_i.size()) == 2 or len(y_i.size()) == 2:
                x_i = x_i.contiguous()
                y_i = y_i.contiguous()
                x_i = x_i.view(-1)
                y_i = y_i.view(-1)
            vx = x_i - torch.mean(x_i)
            vy = y_i - torch.mean(y_i)
            rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 
                2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))))
            x_m = torch.mean(x_i)
            y_m = torch.mean(y_i)
            x_s = torch.std(x_i)
            y_s = torch.std(y_i)
            ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s,
                2) + torch.pow(x_m - y_m, 2))
            cccs += ccc
        return -cccs


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
