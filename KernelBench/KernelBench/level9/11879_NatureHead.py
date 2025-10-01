import torch
import torch.nn as nn
import torch.nn.functional as F


class NatureHead(torch.nn.Module):
    """ DQN Nature 2015 paper
        input: [None, 84, 84, 4]; output: [None, 3136] -> [None, 512];
    """

    def __init__(self, n):
        super(NatureHead, self).__init__()
        self.conv1 = nn.Conv2d(n, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
        self.dense = nn.Linear(32 * 7 * 7, 512)
        self.output_size = 512

    def forward(self, state):
        output = F.relu(self.conv1(state))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = F.relu(self.dense(output.view(-1, 32 * 7 * 7)))
        return output


def get_inputs():
    return [torch.rand([4, 4, 144, 144])]


def get_init_inputs():
    return [[], {'n': 4}]
