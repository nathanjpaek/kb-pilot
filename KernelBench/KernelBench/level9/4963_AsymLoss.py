import torch
import numpy as np
import torch.nn as nn


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))
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
    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp,
            dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp,
            dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn,
            dim=1)), dim=1)
    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)
    return tp, fp, fn


class AsymLoss(nn.Module):

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True,
        smooth=1.0, square=False):
        """
        paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8573779
        """
        super(AsymLoss, self).__init__()
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.beta = 1.5

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)
        weight = self.beta ** 2 / (1 + self.beta ** 2)
        asym = (tp + self.smooth) / (tp + weight * fn + (1 - weight) * fp +
            self.smooth)
        if not self.do_bg:
            if self.batch_dice:
                asym = asym[1:]
            else:
                asym = asym[:, 1:]
        asym = asym.mean()
        return -asym


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
