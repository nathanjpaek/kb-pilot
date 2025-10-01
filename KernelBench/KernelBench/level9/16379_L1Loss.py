import torch
import torch.nn.functional as F
import torch.onnx


class L1Loss(torch.nn.Module):
    """
    L1 loss
    """

    def __init__(self, **kwargs):
        super(L1Loss, self).__init__()
        self.loss_w = kwargs.get('loss_weight', 1)

    def forward(self, preds, gts):
        return F.l1_loss(preds.view(-1), gts.view(-1)) * self.loss_w


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
