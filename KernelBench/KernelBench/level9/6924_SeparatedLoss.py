import torch
import torch.nn as nn


class BCEFocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * (1.0 - probas
            ) ** self.gamma * bce_loss + (1.0 - targets
            ) * probas ** self.gamma * bce_loss
        loss = loss.mean()
        return loss


class SeparatedLoss(nn.Module):

    def __init__(self, loss_disease_risk='BCEWithLogitsLoss', loss_disease=
        'BCEFocalLoss', weights=[1.0, 1.0]):
        super().__init__()
        if loss_disease_risk == 'BCEWithLogitsLoss':
            self.loss_disease_risk = nn.BCEWithLogitsLoss()
        elif loss_disease_risk == 'BCEFocalLoss':
            self.loss_disease_risk = BCEFocalLoss()
        else:
            raise NotImplementedError
        if loss_disease == 'BCEWithLogitsLoss':
            self.loss_disease = nn.BCEWithLogitsLoss()
        elif loss_disease == 'BCEFocalLoss':
            self.loss_disease = BCEFocalLoss()
        self.weights = weights

    def forward(self, preds, targets):
        risk_pred = preds[:, 0]
        risk_targ = targets[:, 0]
        disease_pred = preds[:, 1:]
        disease_targ = targets[:, 1:]
        loss_disease_risk = self.loss_disease_risk(risk_pred, risk_targ)
        loss_disease = self.loss_disease(disease_pred, disease_targ)
        return self.weights[0] * loss_disease_risk + self.weights[1
            ] * loss_disease


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
