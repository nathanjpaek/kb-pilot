import torch
import torch.nn as nn
import torch.nn.functional as F


def adjust_smooth_l1_loss(y_pred, theta=0.1):
    less_grad_factor = 1.0 / (2 * theta)
    less_loss_bias = less_grad_factor * theta ** 2
    less_than_theta = (y_pred < theta).float()
    loss = less_than_theta * y_pred ** 2 * less_grad_factor + (1 -
        less_than_theta) * (y_pred - theta + less_loss_bias)
    return loss


class CrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        """ NLLLoss: negative log likelihood loss.
            # nll_loss: weights: None | a tensor of size C
                        pred in [N, C, d1, d2, ..., dk]
                        target in [N, d1, d2, ..., dk]
                        output in [N, d1, d2, ..., dk]
        """
        super(CrossEntropyLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, ignore_index=ignore_index,
            reduction=reduction)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class BinaryLoss(nn.Module):
    """This class computes the Binary loss to force BG pixels close to 0 and FG pixels far away.
    """

    def __init__(self, margin=2.0, FG_stCH=1, loss_type='l1', weights=None):
        """
        margin: minimum distance between FG/BG if prediction with 1 channel
        FG_stCH: start channel of FG objects on prediction with multiple channels
        loss_type: 'l1' | 'CE', works for prediction with multiple channel.
                   if 'l1', prediction is expected to be softmax2d output.
                   if 'CE', prediction is expected to be net logits
                   if prediction has channel=1,
        weights: if not None, a tensor of size C
        """
        super(BinaryLoss, self).__init__()
        self.margin = margin
        self.FG_stCH = FG_stCH
        self.loss_type = loss_type if FG_stCH > 1 else 'l1'
        if self.loss_type == 'CE':
            self.CE_loss = CrossEntropyLoss(weight=weights, reduction='none')

    def forward(self, preds, targets, weights=None):
        """
        Params:
            preds/targets: [bs, ch, ht, wd]
            weights:[bs, 1, ht, wd]
        """
        _bs, ch, _ht, _wd = preds.size()
        if ch > 1:
            if self.loss_type == 'l1':
                preds_0 = preds[:, :self.FG_stCH, :, :]
                targets_0 = targets[:, :self.FG_stCH, :, :].float()
                loss = adjust_smooth_l1_loss(torch.abs(targets_0 - preds_0))
                loss = loss.sum(axis=1, keepdim=True)
            else:
                preds_0 = preds[:, :self.FG_stCH, :, :].float()
                targets_0 = targets[:, :self.FG_stCH, :, :]
                preds_1, _ = preds[:, self.FG_stCH:, :, :].float().max(axis
                    =1, keepdim=True)
                targets_1 = targets[:, self.FG_stCH:, :, :].sum(axis=1,
                    keepdim=True).int()
                _, target_id = torch.cat([targets_0, targets_1], axis=1).max(
                    axis=1)
                loss = self.CE_loss(torch.cat((preds_0, preds_1), axis=1),
                    target_id)
                loss = loss[:, None, :, :]
        else:
            isFG = (targets > 0.5).float()
            loss_0 = adjust_smooth_l1_loss(F.relu(preds))
            loss_1 = adjust_smooth_l1_loss(F.relu(self.margin - preds))
            loss = loss_0 * (1.0 - isFG) + loss_1 * isFG
        if weights is not None:
            loss = torch.mul(loss, weights).sum() / (weights.sum() + 0.0001)
        else:
            loss = loss.mean()
        return loss.float()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
