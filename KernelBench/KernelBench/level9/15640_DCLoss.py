import torch


def apply_reduction(losses, reduction='none'):
    """Apply reduction to collection of losses."""
    if reduction == 'mean':
        losses = losses.mean()
    elif reduction == 'sum':
        losses = losses.sum()
    return losses


class DCLoss(torch.nn.Module):
    """DC loss function module.

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
        super(DCLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        losses = ((target - input).mean(-1) ** 2).abs() / target.abs().mean(-1
            ) ** 2
        losses = apply_reduction(losses, self.reduction)
        return losses


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
