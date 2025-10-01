import torch
import torch.nn as nn
import torch.nn.functional as F


class Softmax_Policy(nn.Module):
    """
    Simple neural network with softmax action selection
    """

    def __init__(self, num_inputs, hidden_size, action_space):
        super(Softmax_Policy, self).__init__()
        num_outputs = action_space
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return F.softmax(action_scores)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'hidden_size': 4, 'action_space': 4}]
