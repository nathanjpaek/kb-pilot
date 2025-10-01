import torch
import torch.nn as nn


class LogReg(nn.Module):
    """Logreg class."""

    def __init__(self, num_features: 'int', num_classes: 'int'):
        """Initialize the class."""
        super().__init__()
        self.lin_layer = nn.Linear(in_features=num_features, out_features=
            num_classes, bias=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs: 'torch.Tensor', targets: 'torch.Tensor'
        ) ->torch.Tensor:
        """Step forward."""
        out = self.lin_layer(inputs)
        loss = self.criterion(out, targets)
        return loss, out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4, 'num_classes': 4}]
