import torch
import torch.nn as nn
import torch.optim


class EDMLoss(nn.Module):

    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target: 'torch.Tensor', p_estimate: 'torch.Tensor'):
        assert p_target.shape == p_estimate.shape
        cdf_target = torch.cumsum(p_target, dim=1)
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff
            ), 2)))
        return samplewise_emd.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
