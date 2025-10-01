import torch
import torch.nn as nn
import torch.utils.data


class MSEGradLoss(nn.Module):

    def __init__(self, grad=False):
        super(MSEGradLoss, self).__init__()
        self.grad = grad

    def forward(self, input, target):
        err = input - target
        loss = err.norm(p=2).pow(2).div(err.numel())
        if self.grad:
            loss += utils.imGrad(err, bc='reflexive').norm(p=2).pow(2).div(err
                .numel())
        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'gradMSE = ' + str(self.grad
            ) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
