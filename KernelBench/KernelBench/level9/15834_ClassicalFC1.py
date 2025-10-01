import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils.prune
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data


class ClassicalFC1(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        bsz = x.shape[0]
        x = x.view(bsz, 784)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x * x
        output = F.log_softmax(x, dim=1)
        return output.squeeze()


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
