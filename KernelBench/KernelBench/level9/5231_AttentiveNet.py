import torch
from torch import nn
import torch.nn.functional as F


class AttentiveNet(nn.Module):

    def __init__(self, input_size, hidden_size) ->None:
        super().__init__()
        self.cov2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3,
            padding=1)
        self.cov1 = nn.Conv1d(input_size, hidden_size, kernel_size=1, padding=0
            )
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x1 = self.cov1(x)
        x = self.cov2(x1)
        x = x.permute(0, 2, 1).contiguous()
        out = torch.mean(x, dim=1)
        out = F.relu(out)
        out = F.sigmoid(self.dense(out))
        return out.squeeze()


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
