import torch
from torch import nn
import torch.onnx
from torch.nn.parallel.scatter_gather import gather
import torch.utils.data


def _gather_feat(feat, ind, mask=None, trt=False):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    if trt:
        feat = gather(feat, 1, ind)
    else:
        feat = torch.gather(feat, 1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind, trt=False):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind, trt=trt)
    return feat


def _reg_loss(regr, gt_regr, mask):
    """ L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    regr = regr * mask
    gt_regr = gt_regr * mask
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 0.0001)
    return regr_loss


class RegLoss(nn.Module):
    """Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4]), torch.ones([4,
        4], dtype=torch.int64), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
