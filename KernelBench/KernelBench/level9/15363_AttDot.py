import torch
import torch.nn.functional as F


class AttDot(torch.nn.Module):
    """
    AttDot: Dot attention that can be used by the Alignment module.
    """

    def __init__(self, softmax=True):
        super().__init__()
        self.softmax = softmax

    def forward(self, query, y):
        att = torch.bmm(query, y.transpose(2, 1))
        sim = att.max(2)[0].unsqueeze(1)
        if self.softmax:
            att = F.softmax(att, dim=2)
        return att, sim


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
