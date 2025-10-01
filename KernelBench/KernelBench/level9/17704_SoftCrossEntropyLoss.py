import torch
import torch.nn as nn


class SoftCrossEntropyLoss(nn.Module):
    """Cross entropy loss with soft label as target
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth
        =False, batch_average=True):
        super(SoftCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.label_smooth = label_smooth
        self.batch_average = batch_average

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size, num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        if self.use_gpu:
            targets = targets
        if self.label_smooth:
            targets = (1 - self.epsilon
                ) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(1)
        if self.batch_average:
            loss = loss.mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_classes': 4}]
