import torch
import torch.cuda
import torch.nn.functional as F
import torch.nn
import torch.utils.data
import torch.fx
import torch.utils.tensorboard._pytorch_graph


class _TestNetStrided(torch.nn.Module):

    def __init__(self):
        super(_TestNetStrided, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(20, 50, kernel_size=5, stride=(2, 2))
        self.fc1 = torch.nn.Linear(200, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 200)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
