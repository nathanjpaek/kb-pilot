from torch.autograd import Function
import torch


class PairwiseDistance(Function):

    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 0.0001 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1.0 / self.norm)


class TripletMarginLoss(torch.nn.Module):
    """Triplet loss function.
    """

    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)

    def forward(self, repr):
        queue_len_plus = repr.shape[-1]
        pos = repr[:, 0].unsqueeze(-1).repeat(1, queue_len_plus - 1)
        neg = repr[:, 1:]
        dist_hinge = torch.clamp(self.margin + neg - pos, min=0.0)
        loss = torch.mean(dist_hinge, 1).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'margin': 4}]
