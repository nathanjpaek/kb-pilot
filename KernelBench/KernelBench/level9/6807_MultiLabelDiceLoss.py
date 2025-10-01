import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """Differentiable soft dice loss.

    Note: Sigmoid is automatically applied here!
    """

    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        eps = 1e-09
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1).float()
        intersection = torch.sum(m1 * m2, 1)
        union = torch.sum(m1, dim=1) + torch.sum(m2, dim=1)
        score = (2 * intersection + eps) / (union + eps)
        score = (1 - score).mean()
        return score


class MultiLabelDiceLoss(nn.Module):
    """The average dice across multiple classes.

    Note: Sigmoid is automatically applied here!
    """

    def __init__(self):
        super(MultiLabelDiceLoss, self).__init__()
        self.dice_loss = SoftDiceLoss()

    def forward(self, logits, targets):
        loss = 0
        num_classes = targets.size(1)
        for class_nr in range(num_classes):
            loss += self.dice_loss(logits[:, class_nr, :, :], targets[:,
                class_nr, :, :])
        return loss / num_classes


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
