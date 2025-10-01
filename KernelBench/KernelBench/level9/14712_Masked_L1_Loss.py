import torch
import torch.nn as nn


def check_loss_input(im0, im1, w):
    """ im0 is out and im1 is target and w is mask"""
    assert list(im0.size())[2:] == list(im1.size())[2:], 'spatial dim mismatch'
    if w is not None:
        assert list(im0.size())[2:] == list(w.size())[2:
            ], 'spatial dim mismatch'
    if im1.size(0) != 1:
        assert im0.size(0) == im1.size(0)
    if w is not None and w.size(0) != 1:
        assert im0.size(0) == w.size(0)
    return


class Masked_L1_Loss(nn.Module):

    def __init__(self):
        super(Masked_L1_Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, pred, ref, w=None):
        """ ims have dimension BCHW while mask is B1HW """
        check_loss_input(pred, ref, w)
        loss = self.loss(pred, ref)
        assert pred.shape[1] == ref.shape[1]
        channels = pred.shape[1]
        if w is not None:
            w = w.repeat(1, channels, 1, 1)
            n = torch.sum(loss * w, [1, 2, 3])
            d = torch.sum(w, [1, 2, 3])
            loss = n / d
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
