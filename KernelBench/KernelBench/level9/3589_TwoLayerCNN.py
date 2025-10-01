import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerCNN(nn.Module):

    def __init__(self, C, M, embedding, channel, mtc_input, *args, **kwargs):
        super(TwoLayerCNN, self).__init__()
        self.C = C
        self.M = M
        self.embedding = embedding
        self.mtc_input = C if mtc_input else 1
        self.conv1 = nn.Conv1d(self.mtc_input, channel, 3, 1, padding=1,
            bias=False)
        self.flat_size = M // 2 * C // self.mtc_input * channel
        self.fc1 = nn.Linear(self.flat_size, embedding)

    def forward(self, x):
        N = len(x)
        x = x.view(-1, self.mtc_input, self.M)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = x.view(N, self.flat_size)
        x = self.fc1(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'C': 4, 'M': 4, 'embedding': 4, 'channel': 4, 'mtc_input': 4}]
