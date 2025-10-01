import torch
from abc import ABC
from sklearn.preprocessing import Binarizer


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


class Binarizer(BaseOperator, torch.nn.Module):
    """
    Class implementing Binarizer operators in PyTorch.
    """

    def __init__(self, threshold, device):
        super(Binarizer, self).__init__()
        self.transformer = True
        self.threshold = torch.nn.Parameter(torch.FloatTensor([threshold]),
            requires_grad=False)

    def forward(self, x):
        return torch.gt(x, self.threshold).float()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'threshold': 4, 'device': 0}]
