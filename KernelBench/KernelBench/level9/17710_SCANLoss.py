import torch
from torch import nn
import torch.nn.functional as F


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    if input_as_probabilities:
        x_ = torch.clamp(x, min=1e-08)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    if len(b.size()) == 2:
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:
        return -b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % len(b.size()))


class SCANLoss(nn.Module):

    def __init__(self, entropy_weight=2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]
        output:
            - Loss
        """
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.
            view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        entropy_loss = entropy(torch.mean(anchors_prob, 0),
            input_as_probabilities=True)
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        return total_loss, consistency_loss, entropy_loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 1])]


def get_init_inputs():
    return [[], {}]
