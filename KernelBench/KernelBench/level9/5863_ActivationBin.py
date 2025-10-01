from torch.autograd import Function
import torch
import torch.nn as nn


class BinaryActivation(Function):

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1.0)] = 0
        grad_input[input.le(-1.0)] = 0
        """
        #******************soft_ste*****************
        size = input.size()
        zeros = torch.zeros(size).cuda()
        grad = torch.max(zeros, 1 - torch.abs(input))
        #print(grad)
        grad_input = grad_output * grad
        """
        return grad_input


class ActivationBin(nn.Module):

    def __init__(self, A):
        super(ActivationBin, self).__init__()
        self.A = A
        self.relu = nn.ReLU(inplace=True)

    def binary(self, input):
        output = BinaryActivation.apply(input)
        return output

    def forward(self, input):
        if self.A == 2:
            output = self.binary(input)
        else:
            output = self.relu(input)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'A': 4}]
