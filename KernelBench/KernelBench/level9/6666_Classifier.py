import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(900, 3)

    def forward(self, x):
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, 900)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 121, 121])]


def get_init_inputs():
    return [[], {}]
