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


class Scaler(BaseOperator, torch.nn.Module):
    """
    Class implementing Scaler operators in PyTorch. Supported normalizers are L1, L2 and Max.
    """

    def __init__(self, offset, scale, device):
        super(Scaler, self).__init__(transformer=True)
        self.offset = offset
        self.scale = scale
        if offset is not None:
            self.offset = torch.nn.Parameter(torch.DoubleTensor([offset]),
                requires_grad=False)
        if scale is not None:
            self.scale = torch.nn.Parameter(torch.DoubleTensor([scale]),
                requires_grad=False)

    def forward(self, x):
        if self.offset is not None:
            x = x - self.offset
        if self.scale is not None:
            x = x * self.scale
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'offset': 4, 'scale': 1.0, 'device': 0}]
