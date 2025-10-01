import torch
import torch.nn as nn
import torch.onnx


def _neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
    neg_weights = torch.pow(1 - gt[neg_inds], 4)
    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2
            ) * neg_weights
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _regr_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)
    regr = regr[mask]
    gt_regr = gt_regr[mask]
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 0.0001)
    return regr_loss


def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=0.0001, max=1 - 0.0001)
    return x


class CTLoss(nn.Module):

    def __init__(self, regr_weight=1, focal_loss=_neg_loss):
        super(CTLoss, self).__init__()
        self.regr_weight = regr_weight
        self.focal_loss = focal_loss
        self.regr_loss = _regr_loss

    def forward(self, outs, targets):
        stride = 9
        t_heats = outs[0::stride]
        l_heats = outs[1::stride]
        b_heats = outs[2::stride]
        r_heats = outs[3::stride]
        ct_heats = outs[4::stride]
        t_regrs = outs[5::stride]
        l_regrs = outs[6::stride]
        b_regrs = outs[7::stride]
        r_regrs = outs[8::stride]
        gt_t_heat = targets[0]
        gt_l_heat = targets[1]
        gt_b_heat = targets[2]
        gt_r_heat = targets[3]
        gt_ct_heat = targets[4]
        gt_mask = targets[5]
        gt_t_regr = targets[6]
        gt_l_regr = targets[7]
        gt_b_regr = targets[8]
        gt_r_regr = targets[9]
        focal_loss = 0
        t_heats = [_sigmoid(t) for t in t_heats]
        l_heats = [_sigmoid(l) for l in l_heats]
        b_heats = [_sigmoid(b) for b in b_heats]
        r_heats = [_sigmoid(r) for r in r_heats]
        ct_heats = [_sigmoid(ct) for ct in ct_heats]
        focal_loss += self.focal_loss(t_heats, gt_t_heat)
        focal_loss += self.focal_loss(l_heats, gt_l_heat)
        focal_loss += self.focal_loss(b_heats, gt_b_heat)
        focal_loss += self.focal_loss(r_heats, gt_r_heat)
        focal_loss += self.focal_loss(ct_heats, gt_ct_heat)
        regr_loss = 0
        for t_regr, l_regr, b_regr, r_regr in zip(t_regrs, l_regrs, b_regrs,
            r_regrs):
            regr_loss += self.regr_loss(t_regr, gt_t_regr, gt_mask)
            regr_loss += self.regr_loss(l_regr, gt_l_regr, gt_mask)
            regr_loss += self.regr_loss(b_regr, gt_b_regr, gt_mask)
            regr_loss += self.regr_loss(r_regr, gt_r_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss
        loss = (focal_loss + regr_loss) / len(t_heats)
        return loss.unsqueeze(0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([10, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
