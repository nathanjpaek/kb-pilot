import torch
from torch.nn.modules.loss import _Loss


def of_l1_loss(pred_ofsts, kp_targ_ofst, labels, sigma=1.0, normalize=True,
    reduce=False):
    """
    :param pred_ofsts:      [bs, n_kpts, n_pts, c]
    :param kp_targ_ofst:    [bs, n_pts, n_kpts, c]
    :param labels:          [bs, n_pts, 1]
    """
    w = (labels > 1e-08).float()
    bs, n_kpts, n_pts, c = pred_ofsts.size()
    sigma ** 3
    w = w.view(bs, 1, n_pts, 1).repeat(1, n_kpts, 1, 1).contiguous()
    kp_targ_ofst = kp_targ_ofst.view(bs, n_pts, n_kpts, c)
    kp_targ_ofst = kp_targ_ofst.permute(0, 2, 1, 3).contiguous()
    diff = pred_ofsts - kp_targ_ofst
    abs_diff = torch.abs(diff)
    abs_diff = w * abs_diff
    in_loss = abs_diff
    if normalize:
        in_loss = torch.sum(in_loss.view(bs, n_kpts, -1), 2) / (torch.sum(w
            .view(bs, n_kpts, -1), 2) + 0.001)
    if reduce:
        in_loss = torch.mean(in_loss)
    return in_loss


class OFLoss(_Loss):

    def __init__(self):
        super(OFLoss, self).__init__(True)

    def forward(self, pred_ofsts, kp_targ_ofst, labels, normalize=True,
        reduce=False):
        l1_loss = of_l1_loss(pred_ofsts, kp_targ_ofst, labels, sigma=1.0,
            normalize=True, reduce=False)
        return l1_loss


def get_inputs():
    return [torch.rand([4, 1, 4, 1]), torch.rand([4, 1, 4, 1]), torch.rand(
        [4, 1, 4, 1])]


def get_init_inputs():
    return [[], {}]
