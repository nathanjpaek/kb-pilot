import torch
import numpy as np
from torch import nn
import torch.jit
import torch.nn.functional as F
import torch.nn.functional


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
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
    tn = (1 - net_output) * (1 - y_onehot)
    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp,
            dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp,
            dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn,
            dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn,
            dim=1)), dim=1)
    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2
    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)
    return tp, fp, fn, tn


def soft_erode(I):
    p1 = -F.max_pool3d(-I, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    p2 = -F.max_pool3d(-I, (1, 3, 1), (1, 1, 1), (0, 1, 0))
    p3 = -F.max_pool3d(-I, (1, 1, 3), (1, 1, 1), (0, 0, 1))
    return torch.min(torch.min(p1, p3), p2)


def soft_dilate(I):
    return F.max_pool3d(I, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(I):
    return soft_dilate(soft_erode(I))


def soft_skel(img, k=50):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for iter in range(k):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    if torch.cuda.is_available():
        del img1
        del delta
    return skel


class SoftClDiceLoss(nn.Module):

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True,
        smooth=1.0, k=2):
        """
        """
        super(SoftClDiceLoss, self).__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.k = k

    def softCenterline(self, I):
        max = nn.MaxPool3d(3, stride=1, padding=1)
        relu = nn.ReLU()
        Ip = max(-max(-I))
        cl = relu(I - Ip)
        for iter in range(self.k):
            I = -max(-I)
            Ip = max(-max(-I))
            cl = cl + cl * relu(I - Ip)
        return cl

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        clp = soft_skel(x)
        cll = soft_skel(y)
        tp, _fp, fn, _tn = get_tp_fp_fn_tn(x, cll, axes, loss_mask, False)
        tpc, fpc, _fnc, _tnc = get_tp_fp_fn_tn(clp, y, axes, loss_mask, False)
        clp2vollnom = tpc + self.smooth
        clp2vollden = tpc + fpc + self.smooth
        clp2voll = clp2vollnom / clp2vollden
        cll2volpnom = tp + self.smooth
        cll2volpden = tp + fn + self.smooth
        cll2volp = cll2volpnom / cll2volpden
        dc = 2 * clp2voll * cll2volp / (cll2volp + clp2voll + 1e-08)
        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()
        return 1 - dc


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
