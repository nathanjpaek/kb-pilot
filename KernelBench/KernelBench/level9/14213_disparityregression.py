from _paritybench_helpers import _mock_config
import torch
import numpy as np
from torch import nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.parallel
import torch.utils.data.distributed


class disparityregression(nn.Module):

    def __init__(self, maxdisp, cfg):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.array(range(maxdisp))),
            requires_grad=False)

    def forward(self, x, depth):
        out = torch.sum(x * depth[None, :, None, None], 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'maxdisp': 4, 'cfg': _mock_config()}]
