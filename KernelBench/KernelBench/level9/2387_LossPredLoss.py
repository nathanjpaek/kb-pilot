import torch
import torch.nn as nn


class LossPredLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred_loss, target_loss):
        pred_loss = (pred_loss - pred_loss.flip(0))[:len(pred_loss) // 2]
        target_loss = (target_loss - target_loss.flip(0))[:len(target_loss) //
            2]
        target_loss = target_loss.detach()
        one = 2 * torch.sign(torch.clamp(target_loss, min=0)) - 1
        loss = torch.sum(torch.clamp(1.0 - one * pred_loss, min=0))
        loss = loss / pred_loss.size(0)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
