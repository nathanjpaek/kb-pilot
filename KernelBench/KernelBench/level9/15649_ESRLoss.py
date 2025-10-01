import torch


def apply_reduction(losses, reduction='none'):
    """Apply reduction to collection of losses."""
    if reduction == 'mean':
        losses = losses.mean()
    elif reduction == 'sum':
        losses = losses.sum()
    return losses


class ESRLoss(torch.nn.Module):
    """Error-to-signal ratio loss function module.

    See [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of elements in the output,
        'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, reduction='mean'):
        super(ESRLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        losses = ((target - input).abs() ** 2).sum(-1) / (target.abs() ** 2
            ).sum(-1)
        losses = apply_reduction(losses, reduction=self.reduction)
        return losses


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
