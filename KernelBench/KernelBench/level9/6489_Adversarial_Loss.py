import torch
import torch.nn as nn
from numpy import *


class Adversarial_Loss(nn.Module):

    def __init__(self, lambda_adv):
        super(Adversarial_Loss, self).__init__()
        self.lambda_adv = lambda_adv
        pass

    def forward(self, input_p, input_h):
        dis_p = input_p * torch.log(input_p)
        dis_h = torch.log(torch.ones_like(input_h) - input_h)
        adv_loss = dis_h + dis_p
        return torch.sum(self.lambda_adv * adv_loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lambda_adv': 4}]
