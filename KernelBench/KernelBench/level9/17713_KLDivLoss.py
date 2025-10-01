import torch


class KLDivLoss(torch.nn.KLDivLoss):

    def __init__(self, reduction='none'):
        super().__init__(reduction=reduction)

    def forward(self, preds, targets):
        """
        Applies ``log_softmax`` to ``pred`` and ``softmax`` to ``targets``
        prior to computing KL-Divergence loss. These operations are performed
        due to the requirements of the PyTorch API for KLDivLoss.
        """
        preds_ = torch.nn.functional.log_softmax(preds, dim=1)
        targets_ = torch.nn.functional.softmax(targets, dim=1)
        return super(KLDivLoss, self).forward(preds_, targets_)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
