import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class MLP3_clamp_train(nn.Module):
    """
    For unary training, activation clamp is better to be after relu.
    no difference for inference whether clamp is after or before relu.
    """

    def __init__(self):
        super(MLP3_clamp_train, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 512)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32)
        x = F.relu(self.fc1(x)).clamp(-1, 1)
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x)).clamp(-1, 1)
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)


def get_inputs():
    return [torch.rand([4, 1024])]


def get_init_inputs():
    return [[], {}]
