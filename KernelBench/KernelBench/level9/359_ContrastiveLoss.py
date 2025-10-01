import torch


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_anchor_positive = (anchor - positive).pow(2).sum(1)
        distance_anchor_negative = (anchor - negative).pow(2).sum(1)
        loss_triplet = torch.relu(distance_anchor_positive -
            distance_anchor_negative + self.margin)
        return loss_triplet.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
