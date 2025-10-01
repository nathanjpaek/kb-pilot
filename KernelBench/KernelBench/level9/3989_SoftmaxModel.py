import torch
import torch.nn as nn


class SoftmaxModel(nn.Module):
    """
        Model architecture from:
        https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/
    """

    def __init__(self, num_in, num_hidden, num_out, inplace=False):
        super().__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_out = num_out
        self.lin1 = nn.Linear(num_in, num_hidden)
        self.lin2 = nn.Linear(num_hidden, num_hidden)
        self.lin3 = nn.Linear(num_hidden, num_out)
        self.relu1 = nn.ReLU(inplace=inplace)
        self.relu2 = nn.ReLU(inplace=inplace)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        lin1 = self.relu1(self.lin1(input))
        lin2 = self.relu2(self.lin2(lin1))
        lin3 = self.lin3(lin2)
        return self.softmax(lin3)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_in': 4, 'num_hidden': 4, 'num_out': 4}]
