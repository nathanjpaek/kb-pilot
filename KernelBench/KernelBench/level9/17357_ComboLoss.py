import torch
import torch.nn as nn


def l1_loss(A_tensors, B_tensors):
    return torch.abs(A_tensors - B_tensors)


class ComboLoss(nn.Module):

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, from_logits=True, **
        kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.from_logits = from_logits
        None
        self.loss_classification = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred, y_true, features_single=None, y_pred_tiles=
        None, features_tiles=None, y_pred_tiled_flatten=None):
        loss_ = self.alpha * self.loss_classification(y_pred, y_true).mean()
        if features_tiles is not None and self.beta > 0:
            logits_reconstruction = y_pred_tiles
            loss_tiles_class_ = self.loss_classification(logits_reconstruction,
                y_true).mean()
            loss_ = loss_ + self.beta * loss_tiles_class_
        if (features_single is not None and features_tiles is not None and 
            self.gamma > 0):
            loss_reconstruction_ = l1_loss(features_single, features_tiles
                ).mean()
            loss_ = loss_ + self.gamma * loss_reconstruction_
        return loss_


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
