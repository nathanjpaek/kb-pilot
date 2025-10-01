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


def _offset_loss(offset, gt_offset):
    offset_loss = nn.functional.smooth_l1_loss(offset, gt_offset,
        size_average=True)
    return offset_loss


class fully_connected(nn.Module):

    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn
        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn = self.bn(linear) if self.with_bn else linear
        relu = self.relu(bn)
        return relu


class cls(nn.Module):

    def __init__(self, k, inp_dim, out_dim, cat_num, stride=1, with_bn=True):
        super(cls, self).__init__()
        pad = (k - 1) // 2
        self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad),
            stride=(stride, stride), bias=not with_bn)
        self.bn1 = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2, padding=(pad, pad))
        self.conv2 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad),
            stride=(stride, stride), bias=not with_bn)
        self.bn2 = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.pool2 = nn.MaxPool2d(2, 2, padding=(pad, pad))
        self.final = fully_connected(out_dim, cat_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = torch.max(x, 2)[0]
        x = torch.max(x, 2)[0]
        final = self.final(x)
        return final


class offset(nn.Module):

    def __init__(self, k, inp_dim, out_dim, cat_num, stride=1, with_bn=True):
        super(offset, self).__init__()
        pad = (k - 1) // 2
        self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad),
            stride=(stride, stride), bias=not with_bn)
        self.bn1 = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(2, 2, padding=(pad, pad))
        self.conv2 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad),
            stride=(stride, stride), bias=not with_bn)
        self.bn2 = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.pool2 = nn.AvgPool2d(2, 2, padding=(pad, pad))
        self.final = fully_connected(out_dim, cat_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = torch.max(x, 2)[0]
        x = torch.max(x, 2)[0]
        final = self.final(x)
        return final


class AELossPureCls(nn.Module):

    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1,
        focal_loss=_neg_loss, lamda=4, lamdb=2):
        super(AELossPureCls, self).__init__()
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss = focal_loss
        self.ae_loss = _ae_loss
        self.regr_loss = _regr_loss
        self.lamda = lamda
        self.lamdb = lamdb
        self.cls_loss = nn.CrossEntropyLoss(size_average=True)
        self.offset_loss = _offset_loss

    def forward(self, outs, targets):
        stride = 4
        tl_heats = outs[0:-2:stride]
        br_heats = outs[1:-2:stride]
        tl_regrs = outs[2:-2:stride]
        br_regrs = outs[3:-2:stride]
        cls = outs[-2]
        offset = outs[-1]
        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_mask = targets[2]
        gt_tl_regr = targets[3]
        gt_br_regr = targets[4]
        gt_cls = targets[5]
        gt_offset = targets[6]
        focal_loss = 0
        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]
        focal_loss += self.focal_loss(tl_heats, gt_tl_heat, self.lamda,
            self.lamdb)
        focal_loss += self.focal_loss(br_heats, gt_br_heat, self.lamda,
            self.lamdb)
        regr_loss = 0
        for tl_regr, br_regr in zip(tl_regrs, br_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss
        cls_loss = self.cls_loss(cls, gt_cls)
        cls_loss = self.regr_weight * cls_loss
        offset_loss = self.offset_loss(offset, gt_offset)
        offset_loss = self.regr_weight * offset_loss
        loss = (focal_loss + regr_loss) / len(tl_heats
            ) + cls_loss + offset_loss
        return loss.unsqueeze(0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([7, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
