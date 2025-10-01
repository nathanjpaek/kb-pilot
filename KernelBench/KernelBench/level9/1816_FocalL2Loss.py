import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torch.multiprocessing


class FocalL2Loss(nn.Module):
    """
    Compute focal l2 loss between predict and groundtruth
    :param thre: the threshold to distinguish between the foreground
                 heatmap pixels and the background heatmap pixels
    :param alpha beta: compensation factors to reduce the punishment of easy
                 samples (both easy foreground pixels and easy background pixels) 
    """

    def __init__(self, thre=0.01, alpha=0.1, beta=0.02):
        super().__init__()
        self.thre = thre
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        st = torch.where(torch.ge(gt, self.thre), pred - self.alpha, 1 -
            pred - self.beta)
        factor = torch.abs(1.0 - st)
        loss = (pred - gt) ** 2 * factor * mask
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
