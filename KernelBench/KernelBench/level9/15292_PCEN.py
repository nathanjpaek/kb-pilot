import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.quantization
import torch.utils.data.distributed


class PCEN(nn.Module):

    def __init__(self):
        super(PCEN, self).__init__()
        """
        initialising the layer param with the best parametrised values i searched on web (scipy using theese values)
        alpha = 0.98
        delta=2
        r=0.5
        """
        self.log_alpha = Parameter(torch.FloatTensor([0.98]))
        self.log_delta = Parameter(torch.FloatTensor([2]))
        self.log_r = Parameter(torch.FloatTensor([0.5]))
        self.eps = 1e-06

    def forward(self, x, smoother):
        smooth = (self.eps + smoother) ** -self.log_alpha
        pcen = (x * smooth + self.log_delta
            ) ** self.log_r - self.log_delta ** self.log_r
        return pcen


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
