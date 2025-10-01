import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class MLP3_clamp_eval(nn.Module):

    def __init__(self):
        super(MLP3_clamp_eval, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.fc1_out = torch.zeros(1)
        self.relu1_out = torch.zeros(1)
        self.fc2_out = torch.zeros(1)
        self.relu2_out = torch.zeros(1)
        self.fc3_out = torch.zeros(1)

    def forward(self, x):
        x = x.view(-1, 32 * 32)
        self.fc1_out = self.fc1(x).clamp(-1, 1)
        self.relu1_out = F.relu(self.fc1_out)
        self.fc2_out = self.fc2(self.relu1_out).clamp(-1, 1)
        self.relu2_out = F.relu(self.fc2_out)
        self.fc3_out = self.fc3(self.relu2_out).clamp(-1, 1)
        return F.softmax(self.fc3_out, dim=1)


def get_inputs():
    return [torch.rand([4, 1024])]


def get_init_inputs():
    return [[], {}]
