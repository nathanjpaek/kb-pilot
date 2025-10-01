import torch
import torch.nn as nn
import torch.cuda.amp as amp


class SoftDiceLossV2Func(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, logits, labels, p, smooth):
        logits = logits.float()
        probs = torch.sigmoid(logits)
        numer = 2 * (probs * labels).sum(dim=(1, 2)) + smooth
        denor = (probs.pow(p) + labels).sum(dim=(1, 2)) + smooth
        loss = 1.0 - numer / denor
        ctx.vars = probs, labels, numer, denor, p, smooth
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient of soft-dice loss
        """
        probs, labels, numer, denor, p, _smooth = ctx.vars
        M = numer.view(-1, 1, 1) - (probs * labels).mul_(2)
        N = denor.view(-1, 1, 1) - probs.pow(p)
        mppi_1 = probs.pow(p - 1).mul_(p).mul_(M)
        grads = torch.where(labels == 1, probs.pow(p).mul_(2 * (1.0 - p)) -
            mppi_1 + N.mul_(2), -mppi_1)
        grads = grads.div_((probs.pow(p) + N).pow(2)).mul_(probs).mul_(1.0 -
            probs)
        grads = grads.mul_(grad_output.view(-1, 1, 1)).neg_()
        return grads, None, None, None


class SoftDiceLossV2(nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self, p=1, smooth=1, reduction='mean'):
        super(SoftDiceLossV2, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        args: logits: tensor of shape (N, H, W)
        args: label: tensor of shape(N, H, W)
        """
        loss = SoftDiceLossV2Func.apply(logits, labels, self.p, self.smooth)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
