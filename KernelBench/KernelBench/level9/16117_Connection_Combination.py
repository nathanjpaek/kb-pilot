import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed


class Connection_Combination(nn.Module):
    """combine 3 types of connection method by 'beta' weights to become an input node """

    def __init__(self):
        super(Connection_Combination, self).__init__()

    def forward(self, prev_parallel, prev_above, prev_below, betas):
        betas = F.softmax(betas, dim=-1)
        mix = 3 * betas[0] * prev_parallel + 3 * betas[1
            ] * prev_above + 3 * betas[2] * prev_below
        mix = F.relu(mix)
        return mix


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
