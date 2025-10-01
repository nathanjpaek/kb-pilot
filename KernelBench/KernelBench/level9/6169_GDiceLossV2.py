import torch
import torch.nn as nn
import torch.utils.data
import torch
from torch.autograd import Variable


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order).contiguous()
    return transposed.view(C, -1)


class GDiceLossV2(nn.Module):

    def __init__(self, apply_nonlin=None, smooth=1e-05):
        """
        Generalized Dice;
        Copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLossV2, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape
        shp_y = gt.shape
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))
            if all([(i == j) for i, j in zip(net_output.shape, gt.shape)]):
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == 'cuda':
                    y_onehot = y_onehot
                y_onehot.scatter_(1, gt, 1)
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
        input = flatten(net_output)
        target = flatten(y_onehot)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1.0 / (target_sum * target_sum).clamp(min=
            self.smooth), requires_grad=False)
        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()
        denominator = ((input + target).sum(-1) * class_weights).sum()
        return -2.0 * intersect / denominator.clamp(min=self.smooth)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
