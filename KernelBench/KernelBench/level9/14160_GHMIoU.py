import torch
import torch.nn.functional as F
import torch.nn as nn


class GHMIoU(nn.Module):
    """GHM IoU prediction loss

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0):
        super(GHMIoU, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float() / bins
        self.edges[-1] += 1e-06
        if momentum > 0:
            self.acc_sum = torch.zeros(bins)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch*A*width*height]):
                IoU prediction for each regression anchor
            target (float tensor of size [batch*A*width*height]):
                store the iou between predicted boxes and its corresponding
                groundtruth boxes for the positives and the iou between the
                predicted boxes and anchors for negatives.
            label_weight (float tensor of size [[batch*A*width*height]):
                1 for positives and 0 for others.
        Returns:
            The gradient harmonized loss.
        """
        target, label_weight = target.float(), label_weight.float()
        target = target.detach()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)
        g = torch.abs(pred.sigmoid().detach() - target)
        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt
                        ) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n
        loss = F.binary_cross_entropy_with_logits(pred, target, weights,
            reduction='sum') / tot
        return loss * self.loss_weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
