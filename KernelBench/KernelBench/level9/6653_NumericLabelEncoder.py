import torch
from abc import ABC


class BaseOperator(ABC):
    """
    Abstract class defining the basic structure for operator implementations in Hummingbird.
    """

    def __init__(self, regression=False, classification=False, transformer=
        False, anomaly_detection=False, **kwargs):
        """
        Args:
            regression: Whether operator is a regression model.
            classification: Whether the operator is a classification model.
            transformer: Whether the operator is a feature transformer.
            anomaly_detection: Whether the operator is an anomaly detection model.
            kwargs: Other keyword arguments.
        """
        super().__init__()
        self.regression = regression
        self.classification = classification
        self.transformer = transformer
        self.anomaly_detection = anomaly_detection


class NumericLabelEncoder(BaseOperator, torch.nn.Module):

    def __init__(self, classes, device):
        super(NumericLabelEncoder, self).__init__(transformer=True)
        self.regression = False
        self.check_tensor = torch.nn.Parameter(torch.IntTensor(classes),
            requires_grad=False)

    def forward(self, x):
        x = x.view(-1, 1)
        return torch.argmax(torch.eq(x, self.check_tensor).int(), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'classes': 4, 'device': 0}]
