import torch
import torch.utils.data
import torch.nn as nn


class CharbonnierPenalty(nn.Module):

    def __init__(self, n=0.001, total_variation=False, lam=1e-06, per_pixel
        =False):
        super().__init__()
        self.n = n
        self.total_variation = total_variation
        self.lam = lam
        self.per_pixel = per_pixel

    def forward(self, output, gt):
        assert output.shape == gt.shape, 'output and gt shapes do not match'
        x = output.sub(gt)
        loss = torch.sqrt(x * x + self.n * self.n)
        if self.total_variation:
            loss += self.lam * (torch.sum(torch.abs(x[:, :, :, :-1] - x[:,
                :, :, 1:])) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :,
                1:, :])) + torch.sum(torch.abs(x[:, :-1, :, :] - x[:, 1:, :,
                :])))
        loss = loss.mean() if self.per_pixel else loss.sum() / output.shape[0]
        return loss

    def __repr__(self):
        lmbda = '' if not self.total_variation else ', lambda=' + str(self.lam)
        return '{}_v3(n={}, total_variation={}'.format(self.__class__.
            __name__, self.n, self.total_variation
            ) + lmbda + ', per_pixel=' + str(self.per_pixel) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
