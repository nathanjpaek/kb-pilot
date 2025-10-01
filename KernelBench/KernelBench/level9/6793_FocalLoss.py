import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalSigmoidLossFunc(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    def forward(ctx, logits, label, alpha, gamma):
        logits = logits.float()
        coeff = torch.empty_like(logits).fill_(1 - alpha)
        coeff[label == 1] = alpha
        probs = torch.sigmoid(logits)
        log_probs = torch.where(logits >= 0, F.softplus(logits, -1, 50), 
            logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0, -logits + F.softplus(logits,
            -1, 50), -F.softplus(logits, 1, 50))
        probs_gamma = probs ** gamma
        probs_1_gamma = (1.0 - probs) ** gamma
        ctx.coeff = coeff
        ctx.probs = probs
        ctx.log_probs = log_probs
        ctx.log_1_probs = log_1_probs
        ctx.probs_gamma = probs_gamma
        ctx.probs_1_gamma = probs_1_gamma
        ctx.label = label
        ctx.gamma = gamma
        term1 = probs_1_gamma * log_probs
        term2 = probs_gamma * log_1_probs
        loss = torch.where(label == 1, term1, term2).mul_(coeff).neg_()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        compute gradient of focal loss
        """
        coeff = ctx.coeff
        probs = ctx.probs
        log_probs = ctx.log_probs
        log_1_probs = ctx.log_1_probs
        probs_gamma = ctx.probs_gamma
        probs_1_gamma = ctx.probs_1_gamma
        label = ctx.label
        gamma = ctx.gamma
        term1 = (1.0 - probs - gamma * probs * log_probs).mul_(probs_1_gamma
            ).neg_()
        term2 = (probs - gamma * (1.0 - probs) * log_1_probs).mul_(probs_gamma)
        grads = torch.where(label == 1, term1, term2).mul_(coeff).mul_(
            grad_output)
        return grads, None, None, None


class FocalLoss(nn.Module):
    """
    This use better formula to compute the gradient, which has better numeric stability
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        loss = FocalSigmoidLossFunc.apply(logits, label, self.alpha, self.gamma
            )
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
