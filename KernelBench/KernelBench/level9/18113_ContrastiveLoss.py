import torch


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    reference code: https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 2
        _, y_type = in_types
        assert y_type.dim() == 1, ValueError(y_type.shape)

    def forward(self, dist, y):
        self.check_type_forward((dist, y))
        dist_sq = torch.pow(dist, 2)
        mdist = self.margin - dist_sq
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * dist
        return loss


def get_inputs():
    return [torch.rand([4]), torch.rand([4])]


def get_init_inputs():
    return [[], {}]
