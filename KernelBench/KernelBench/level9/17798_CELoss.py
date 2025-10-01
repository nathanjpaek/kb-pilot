import torch
import torch.nn.functional as F
import torch.nn as nn


class MetricLearningLoss(nn.Module):
    """
    Generic loss function to be used in a metric learning setting
    """

    def __init__(self, embedding_size, n_classes, device='cpu', *args, **kwargs
        ):
        super(MetricLearningLoss, self).__init__()
        self.embedding_size = embedding_size
        self.n_classes = n_classes
        self.device = device

    def forward(self, inputs, targets):
        raise NotImplementedError()


class CELoss(MetricLearningLoss):
    """
    Cross-entropy loss with the addition of a linear layer
    to map inputs to the target number of classes
    """

    def __init__(self, embedding_size, n_classes, device='cpu'):
        super(CELoss, self).__init__(embedding_size, n_classes, device=device)
        self.fc = nn.Linear(embedding_size, n_classes)

    def forward(self, inputs, targets):
        """
        Compute cross-entropy loss for inputs of shape
        [B, E] and targets of size [B]

        B: batch size
        E: embedding size
        """
        logits = self.fc(inputs)
        preds = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, targets)
        inputs = F.normalize(inputs, p=2, dim=1)
        return inputs, preds, loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding_size': 4, 'n_classes': 4}]
