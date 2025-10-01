import torch
import torchvision.transforms.functional as F
import torch.nn.functional as F
import torch.utils.model_zoo


def pdist(A, squared=False, eps=0.0001):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    if squared:
        return res
    else:
        return res.clamp(min=eps).sqrt()


class TripletAllLoss(torch.nn.Module):
    """
    Compute all possible triplets in the batch, where loss > 0
    """

    def __init__(self, margin=1.0, **kwargs):
        None
        self.margin = margin
        torch.nn.Module.__init__(self)

    def forward(self, embeddings, labels):
        d = pdist(embeddings)
        pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]
            ).type_as(d) - torch.eye(len(d)).type_as(d)
        T = d.unsqueeze(1).expand(*((len(d),) * 3))
        M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
        dist_diff = F.relu(T - T.transpose(1, 2) + self.margin)
        return (M * dist_diff).sum() / M.sum()


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4])]


def get_init_inputs():
    return [[], {}]
