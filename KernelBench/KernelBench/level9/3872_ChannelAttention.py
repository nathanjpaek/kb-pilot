import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class ChannelAttention(nn.Module):

    def __init__(self, C):
        super(ChannelAttention, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(C, int(C / 4))
        self.fc2 = nn.Linear(int(C / 4), C)

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, kernel_size=x.size()[-1])
        avg_pool = avg_pool.permute(0, 2, 3, 1)
        fc = self.fc1(avg_pool)
        relu = self.relu(fc)
        fc = self.fc2(relu).permute(0, 3, 1, 2)
        atten = self.sigmoid(fc)
        output = atten * x
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'C': 4}]
