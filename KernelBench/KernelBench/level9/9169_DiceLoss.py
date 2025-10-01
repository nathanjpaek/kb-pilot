import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        """Cacluate dice loss

         Parameters
         ----------
             pred:
                 predictions from the model
             target:
                 ground truth label
         """
        smooth = 1.0
        p_flat = pred.contiguous().view(-1)
        t_flat = target.contiguous().view(-1)
        intersection = (p_flat * t_flat).sum()
        a_sum = torch.sum(p_flat * p_flat)
        b_sum = torch.sum(t_flat * t_flat)
        return 1 - (2.0 * intersection + smooth) / (a_sum + b_sum + smooth)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
