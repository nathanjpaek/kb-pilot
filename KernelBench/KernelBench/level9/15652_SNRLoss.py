import torch


def apply_reduction(losses, reduction='none'):
    """Apply reduction to collection of losses."""
    if reduction == 'mean':
        losses = losses.mean()
    elif reduction == 'sum':
        losses = losses.sum()
    return losses


class SNRLoss(torch.nn.Module):
    """Signal-to-noise ratio loss module.

    Note that this does NOT implement the SDR from
    [Vincent et al., 2006](https://ieeexplore.ieee.org/document/1643671),
    which includes the application of a 512-tap FIR filter.
    """

    def __init__(self, zero_mean=True, eps=1e-08, reduction='mean'):
        super(SNRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean
        res = input - target
        losses = 10 * torch.log10((target ** 2).sum(-1) / ((res ** 2).sum(-
            1) + self.eps) + self.eps)
        losses = apply_reduction(losses, self.reduction)
        return -losses


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
