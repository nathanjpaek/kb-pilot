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


class Multiply(BaseOperator, torch.nn.Module):
    """
    Module used to multiply features in a pipeline by a score.
    """

    def __init__(self, score):
        super(Multiply, self).__init__()
        self.score = score

    def forward(self, x):
        return x * self.score


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'score': 4}]
