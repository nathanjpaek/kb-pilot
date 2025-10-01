import torch
import torch.nn as nn
import torch.nn.functional


class Net(nn.Module):

    def __init__(self, num_inputs=784, num_outputs=10, num_hiddens=256,
        is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.linear_1 = nn.Linear(num_inputs, num_hiddens)
        self.linear_2 = nn.Linear(num_hiddens, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = X.reshape((-1, self.num_inputs))
        H1 = self.relu(self.linear_1(X))
        out = self.linear_2(H1)
        return out


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
