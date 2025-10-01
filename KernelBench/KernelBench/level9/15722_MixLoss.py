import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import filterfalse


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def isnan(x):
    return x != x


def mean(l, ignore_nan=True, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0, eps=1e-07):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        output = torch.sigmoid(output)
        if torch.sum(target) == 0:
            output = 1.0 - output
            target = 1.0 - target
        return 1.0 - (2 * torch.sum(output * target) + self.smooth) / (
            torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-07):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, logit, target):
        prob = torch.sigmoid(logit)
        prob = prob.clamp(self.eps, 1.0 - self.eps)
        loss = -1 * target * torch.log(prob)
        loss = loss * (1 - logit) ** self.gamma
        return loss.sum()


class LovaszHinge(nn.Module):

    def __init__(self, activation=lambda x: F.elu(x, inplace=True) + 1.0,
        per_image=True, ignore=None):
        super(LovaszHinge, self).__init__()
        self.activation = activation
        self.per_image = per_image
        self.ignore = ignore

    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\\infty and +\\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """
        if len(labels) == 0:
            return logits.sum() * 0.0
        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = lovasz_grad(gt_sorted)
        loss = torch.dot(self.activation(errors_sorted), grad)
        return loss

    def forward(self, logits, labels):
        if self.per_image:
            loss = mean(self.lovasz_hinge_flat(*flatten_binary_scores(log.
                unsqueeze(0), lab.unsqueeze(0), self.ignore)) for log, lab in
                zip(logits, labels))
        else:
            loss = self.lovasz_hinge_flat(*flatten_binary_scores(logits,
                labels, self.ignore))
        return loss


class MixLoss(nn.Module):

    def __init__(self, bce_w=1.0, dice_w=0.0, focal_w=0.0, lovasz_w=0.0,
        bce_kwargs={}, dice_kwargs={}, focal_kwargs={}, lovasz_kwargs={}):
        super(MixLoss, self).__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.focal_w = focal_w
        self.lovasz_w = lovasz_w
        self.bce_loss = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dice_loss = DiceLoss(**dice_kwargs)
        self.focal_loss = FocalLoss(**focal_kwargs)
        self.lovasz_loss = LovaszHinge(**lovasz_kwargs)

    def forward(self, output, target):
        loss = 0.0
        if self.bce_w:
            loss += self.bce_w * self.bce_loss(output, target)
        if self.dice_w:
            loss += self.dice_w * self.dice_loss(output, target)
        if self.focal_w:
            loss += self.focal_w * self.focal_loss(output, target)
        if self.lovasz_w:
            loss += self.lovasz_w * self.lovasz_loss(output, target)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
