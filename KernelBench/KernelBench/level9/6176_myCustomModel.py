import logging
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class BaseModel(nn.Module):
    """
    Base class for all models
    All models require an initialization and forward method, and __str__ is what is shown when print(model) is called
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__(
            ) + '\nTrainable parameters: {}'.format(params)


class myCustomModel(BaseModel):

    def __init__(self, num_classes=2):
        super(myCustomModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
