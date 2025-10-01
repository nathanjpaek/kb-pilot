import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """ Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations"""

    def __init__(self, gamma_neg=4, gamma_pos=0, probability_margin=0.05,
        eps=1e-08, label_smooth=0.0):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.label_smooth = label_smooth
        self.clip = probability_margin
        self.eps = eps
        (self.targets) = (self.anti_targets) = (self.xs_pos) = (self.xs_neg
            ) = (self.asymmetric_w) = (self.loss) = None

    def get_last_scale(self):
        return 1.0

    def forward(self, inputs, targets):
        """"
        Parameters
        ----------
        inputs: input logits
        targets: targets (multi-label binarized vector)
        """
        if self.label_smooth > 0:
            targets = targets * (1 - self.label_smooth)
            targets[targets == 0] = self.label_smooth
        self.targets = targets
        self.anti_targets = 1 - targets
        self.xs_pos = torch.sigmoid(inputs)
        self.xs_neg = 1.0 - self.xs_pos
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=
            self.eps)))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg, 
                self.gamma_pos * self.targets + self.gamma_neg * self.
                anti_targets)
            self.loss *= self.asymmetric_w
        return -self.loss.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
