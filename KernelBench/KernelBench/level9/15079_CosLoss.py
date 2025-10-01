import torch
from torch.nn.modules.loss import _Loss


class CosLoss(_Loss):

    def __init__(self, eps=1e-05):
        super(CosLoss, self).__init__(True)
        self.eps = eps

    def forward(self, pred_ofsts, kp_targ_ofst, labels, normalize=True):
        """
        :param pred_ofsts:      [bs, n_kpts, n_pts, c]
        :param kp_targ_ofst:    [bs, n_pts, n_kpts, c]
        :param labels:          [bs, n_pts, 1]
        """
        None
        w = (labels > 1e-08).float()
        bs, n_kpts, n_pts, _c = pred_ofsts.size()
        pred_vec = pred_ofsts / (torch.norm(pred_ofsts, dim=3, keepdim=True
            ) + self.eps)
        None
        w = w.view(bs, 1, n_pts, 1).repeat(1, n_kpts, 1, 1).contiguous()
        kp_targ_ofst = kp_targ_ofst.view(bs, n_pts, n_kpts, 3)
        kp_targ_ofst = kp_targ_ofst.permute(0, 2, 1, 3).contiguous()
        targ_vec = kp_targ_ofst / (torch.norm(kp_targ_ofst, dim=3, keepdim=
            True) + self.eps)
        cos_sim = pred_vec * targ_vec
        in_loss = -1.0 * w * cos_sim
        if normalize:
            in_loss = torch.sum(in_loss.view(bs, n_kpts, -1), 2) / (torch.
                sum(w.view(bs, n_kpts, -1), 2) + 0.001)
        return in_loss


def get_inputs():
    return [torch.rand([4, 1, 4, 1]), torch.rand([4, 4, 1, 3]), torch.rand(
        [4, 1, 4, 1])]


def get_init_inputs():
    return [[], {}]
