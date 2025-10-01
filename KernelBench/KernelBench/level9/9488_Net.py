import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, input_dim, n_classes):
        super(Net, self).__init__()
        self.n_classes = n_classes
        self.fc = nn.Linear(input_dim, 2048)

    def _forward2(self, x):
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

    def forward(self, x):
        means = x.mean(dim=1, keepdim=True)
        stds = x.std(dim=1, keepdim=True)
        normalized_data = (x - means) / (stds + 1e-16)
        x = self._forward2(normalized_data)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'n_classes': 4}]
