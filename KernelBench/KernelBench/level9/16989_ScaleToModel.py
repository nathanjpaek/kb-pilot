import torch
import torch.nn as nn
import torch.cuda
from torch import linalg as linalg


class ScaleToModel(nn.Module):
    """
    This class acts as an adapter module that scales pixel values from the test run domain to the model domain.
    """

    def __init__(self, model_value_range, test_value_range):
        """
        Initializes the scaler module by setting the model domain and test domain value range.

        Args:
            model_value_range (List[float]): The model's value range.
            test_value_range (List[float]): The test run's value range.
        """
        super(ScaleToModel, self).__init__()
        self.m_min, self.m_max = model_value_range
        self.t_min, self.t_max = test_value_range

    def forward(self, img: 'torch.Tensor'):
        """
        Scales the input image from the test run domain to the model domain.

        Args:
            img (torch.Tensor): The image to scale.

        Returns: The scaled image.
        """
        img = (img - self.t_min) / (self.t_max - self.t_min)
        img = img * (self.m_max - self.m_min) + self.m_min
        return img


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'model_value_range': [4, 4], 'test_value_range': [4, 4]}]
