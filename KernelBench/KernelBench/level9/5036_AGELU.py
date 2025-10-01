import math
import torch
import torch.utils.data
import torch.cuda
import torch.utils.checkpoint


def agelu(x):
    SQRT_M2_PI = math.sqrt(2 / math.pi)
    COEFF = 0.044715
    return 0.5 * x * (1.0 + torch.tanh(SQRT_M2_PI * (x + COEFF * torch.pow(
        x, 3))))


class AGELU(torch.nn.Module):

    def forward(self, input):
        return agelu(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
