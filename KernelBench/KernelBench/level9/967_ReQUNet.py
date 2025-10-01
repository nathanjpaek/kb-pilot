import torch
import torch.nn as nn


def MyReQU(x):
    x[x < 0] = 0
    z = x * x
    return z


class ReQUNet(nn.Module):

    def __init__(self):
        super(ReQUNet, self).__init__()
        n_in, n_h, n_out = 4, 64, 3
        self.fc1 = nn.Linear(n_in, n_h, True)
        self.fc2 = nn.Linear(n_h, n_out, True)

    def forward(self, x):
        h = MyReQU(self.fc1(x))
        pred = self.fc2(h)
        soft = nn.Softmax()
        pred_for_acc = soft(pred)
        return pred, pred_for_acc


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
