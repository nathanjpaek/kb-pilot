import torch
import torch.nn as nn
import torch.utils.data


class Softplus(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = torch.log(1 + torch.exp(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * torch.sigmoid(ctx.saved_variables[0])


class CustomSoftplus(nn.Module):

    def forward(self, input_tensor):
        return Softplus.apply(input_tensor)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
