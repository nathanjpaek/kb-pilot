import torch
import torch.nn.functional as F


class AttDistance(torch.nn.Module):
    """
    AttDistance: Distance attention that can be used by the Alignment module.
    """

    def __init__(self, dist_norm=1, weight_norm=1):
        super().__init__()
        self.dist_norm = dist_norm
        self.weight_norm = weight_norm

    def forward(self, query, y):
        att = (query.unsqueeze(1) - y.unsqueeze(2)).abs().pow(self.dist_norm)
        att = att.mean(dim=3).pow(self.weight_norm)
        att = -att.transpose(2, 1)
        sim = att.max(2)[0].unsqueeze(1)
        att = F.softmax(att, dim=2)
        return att, sim


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
