import torch
import torch.nn as nn
import torch.cuda


def _ae_loss(tag0, tag1, mask):
    num = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()
    tag_mean = (tag0 + tag1) / 2
    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 0.0001)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 0.0001)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1
    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 0.0001)
    dist = dist / (num2 + 0.0001)
    dist = dist[mask]
    push = dist.sum()
    return pull, push


def _neg_loss(preds, gt, lamda, lamdb):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
    neg_weights = torch.pow(1 - gt[neg_inds], lamda)
    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, lamdb)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, lamdb
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


class AELossPurePie(nn.Module):

    def __init__(self, lamda, lamdb, regr_weight=1, focal_loss=_neg_loss):
        super(AELossPurePie, self).__init__()
        self.regr_weight = regr_weight
        self.focal_loss = focal_loss
        self.ae_loss = _ae_loss
        self.regr_loss = _regr_loss
        self.lamda = lamda
        self.lamdb = lamdb

    def forward(self, outs, targets):
        stride = 5
        center_heats = outs[0::stride]
        key_heats = outs[1::stride]
        center_regrs = outs[2::stride]
        key_regrs_tl = outs[3::stride]
        key_regrs_br = outs[4::stride]
        gt_center_heat = targets[0]
        gt_key_heat = targets[1]
        gt_mask = targets[2]
        gt_center_regr = targets[3]
        gt_key_regr_tl = targets[4]
        gt_key_regr_br = targets[5]
        focal_loss = 0
        center_heats = [_sigmoid(t) for t in center_heats]
        key_heats = [_sigmoid(b) for b in key_heats]
        focal_loss += self.focal_loss(center_heats, gt_center_heat, self.
            lamda, self.lamdb)
        focal_loss += self.focal_loss(key_heats, gt_key_heat, self.lamda,
            self.lamdb)
        regr_loss = 0
        for center_regr, key_regr_tl, key_regr_br in zip(center_regrs,
            key_regrs_tl, key_regrs_br):
            regr_loss += self.regr_loss(center_regr, gt_center_regr, gt_mask)
            regr_loss += self.regr_loss(key_regr_tl, gt_key_regr_tl, gt_mask
                ) / 2
            regr_loss += self.regr_loss(key_regr_br, gt_key_regr_br, gt_mask
                ) / 2
        regr_loss = self.regr_weight * regr_loss
        loss = (focal_loss + regr_loss) / len(center_heats)
        return loss.unsqueeze(0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([6, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lamda': 4, 'lamdb': 4}]
