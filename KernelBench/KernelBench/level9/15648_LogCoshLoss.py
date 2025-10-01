import torch


def apply_reduction(losses, reduction='none'):
    """Apply reduction to collection of losses."""
    if reduction == 'mean':
        losses = losses.mean()
    elif reduction == 'sum':
        losses = losses.sum()
    return losses


class LogCoshLoss(torch.nn.Module):
    """Log-cosh loss function module.

    See [Chen et al., 2019](https://openreview.net/forum?id=rkglvsC9Ym).

    Args:
        a (float, optional): Smoothness hyperparameter. Smaller is smoother. Default: 1.0
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, a=1.0, eps=1e-08, reduction='mean'):
        super(LogCoshLoss, self).__init__()
        self.a = a
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        losses = (1 / self.a * torch.log(torch.cosh(self.a * (input -
            target)) + self.eps)).mean(-1)
        losses = apply_reduction(losses, self.reduction)
        return losses


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
