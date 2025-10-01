import torch
import torch.nn as nn


class SimpleNormLayer(nn.Module):
    """Simple normalization layer that divides the output of a
    preceding layer by a specified number

    Parameters
    ----------
    normalization_strength: float
        The number with which input is normalized/dived by
    """

    def __init__(self, normalization_strength):
        super(SimpleNormLayer, self).__init__()
        self.normalization_strength = normalization_strength

    def forward(self, input_features):
        """Computes normalized output

        Parameters
        ----------
        input_features: torch.Tensor
            Input tensor of featuers of any shape

        Returns
        -------
        normalized_features: torch.Tensor
            Normalized input features
        """
        return input_features / self.normalization_strength


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'normalization_strength': 4}]
