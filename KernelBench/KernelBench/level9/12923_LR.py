import torch
import torch.nn as nn
import torch.nn.functional as F


class LR(nn.Module):
    """ Logistinc regression
    """

    def __init__(self, input_nc, input_width, input_height, no_classes=10,
        **kwargs):
        super(LR, self).__init__()
        self.fc = nn.Linear(input_nc * input_width * input_height, no_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.fc(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_nc': 4, 'input_width': 4, 'input_height': 4}]
