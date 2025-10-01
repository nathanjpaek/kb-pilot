import torch
import torch.nn as nn
import torch.nn.init


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1)
        ) - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class CapOnlyContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(CapOnlyContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim
        self.max_violation = max_violation

    def forward(self, im, s, ex_s):
        scores = self.sim(im, ex_s)
        scores_orig = self.sim(im, s)
        diagonal = scores_orig.diag().contiguous().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
        return cost_s.sum()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
