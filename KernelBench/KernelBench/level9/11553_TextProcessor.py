import torch
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F


def reset_parameters_util(model):
    pass


class TextProcessor(nn.Module):
    """Processes sentence representations to the correct hidden dimension"""

    def __init__(self, desc_dim, hid_dim):
        super(TextProcessor, self).__init__()
        self.desc_dim = desc_dim
        self.hid_dim = hid_dim
        self.transform = nn.Linear(desc_dim, hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_util(self)

    def forward(self, desc):
        bs, num_classes, desc_dim = desc.size()
        desc = desc.view(-1, desc_dim)
        out = self.transform(desc)
        out = out.view(bs, num_classes, -1)
        return F.relu(out)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'desc_dim': 4, 'hid_dim': 4}]
