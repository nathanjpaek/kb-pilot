import torch


class Dice_Loss(torch.nn.Module):
    """This is a custom Dice Similarity Coefficient loss function that we use
    to the accuracy of the segmentation. it is defined as ;
    DSC = 2 * (pred /intersect label) / (pred /union label) for the losss we use
    1- DSC so gradient descent leads to better outputs."""

    def __init__(self, weight=None, size_average=False):
        super(Dice_Loss, self).__init__()

    def forward(self, pred, label):
        label = label.float()
        smooth = 1.0
        intersection = torch.sum(pred * label)
        union = torch.sum(pred) + torch.sum(label)
        loss = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
