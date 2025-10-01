import torch
import torch.nn.functional as F


class UniformDistributionLoss(torch.nn.Module):
    """
    Implementation of the confusion loss from
    [Simultaneous Deep Transfer Across Domains and Tasks](https://arxiv.org/abs/1510.02192).
    """

    def forward(self, x, *args):
        """"""
        probs = F.log_softmax(x, dim=1)
        avg_probs = torch.mean(probs, dim=1)
        return -torch.mean(avg_probs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
