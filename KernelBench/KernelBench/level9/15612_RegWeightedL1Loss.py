import torch
from torch import nn
import torch.onnx
from torch.nn.parallel.scatter_gather import gather
import torch.nn.functional as F
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


class RegWeightedL1Loss(nn.Module):

    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 0.0001)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.ones(
        [4, 4], dtype=torch.int64), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
