import torch
import torch.nn as nn


class MinibatchStdDev(nn.Module):
    """
    Mini-batch standard deviation module computes the standard deviation of every feature
    vector of a pixel and concatenates the resulting map to the original tensor
    """

    def __init__(self, alpha: 'float'=1e-08) ->None:
        """
        Constructor method
        :param alpha: (float) Small constant for numeric stability
        """
        super(MinibatchStdDev, self).__init__()
        self.alpha = alpha

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """
        Forward pass
        :param input: (Torch Tensor) Input tensor [batch size, channels,, height, width]
        :return: (Torch Tensor) Output tensor [batch size, channels, height, width]
        """
        output = input - torch.mean(input, dim=0, keepdim=True)
        output = torch.sqrt(torch.mean(output ** 2, dim=0, keepdim=False).
            clamp(min=self.alpha))
        output = torch.mean(output).view(1, 1, 1)
        output = output.repeat(input.shape[0], 1, input.shape[2], input.
            shape[3])
        output = torch.cat((input, output), 1)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
