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


class ComboLoss(nn.Module):
    """Weighted classification and segmentation loss.

    Attributes:
        weights (list):
        activation:
        bce: with logits loss
        dice_loss: soft dice loss (all classes)

    """

    def __init__(self, weights=[0.1, 0, 1], activation=None):
        """
        Args:
            weights (list): [image_cls, pixel_seg, pixel_cls]
            activation: One of ['sigmoid', None]
        """
        super(ComboLoss, self).__init__()
        self.weights = weights
        self.activation = activation
        assert self.activation in ['sigmoid', None
            ], "`activation` must be one of ['sigmoid', None]."
        self.bce = nn.BCEWithLogitsLoss(reduce=True)
        self.dice_loss = MultiLabelDiceLoss()

    def create_fc_tensors(self, logits, targets):
        """Creates the classification tensors from the segmentation ones.
        """
        batch_size, num_classes, _, _ = targets.shape
        summed = targets.view(batch_size, num_classes, -1).sum(-1)
        targets_fc = (summed > 0).float()
        logits_fc = logits.view(batch_size, num_classes, -1)
        logits_fc = torch.max(logits_fc, -1)[0]
        return logits_fc, targets_fc

    def forward(self, logits, targets):
        logits_fc, targets_fc = self.create_fc_tensors(logits, targets)
        p = torch.sigmoid(logits) if self.activation == 'sigmoid' else logits
        if self.weights[0]:
            loss_fc = self.weights[0] * self.bce(logits_fc, targets_fc)
        else:
            loss_fc = torch.tensor(0)
        if self.weights[1] or self.weights[2]:
            loss_seg_dice = self.weights[1] * self.dice_loss(p, targets)
            loss_seg_bce = self.weights[2] * self.bce(logits, targets)
        else:
            loss_seg_dice = torch.tensor(0)
            loss_seg_bce = torch.tensor(0)
        loss = loss_fc + loss_seg_bce + loss_seg_dice
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
