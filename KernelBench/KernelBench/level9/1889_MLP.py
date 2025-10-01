import torch
import torch.nn.functional as F
import torch.utils.data


class MLP(torch.nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(784, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, din):
        din = din.view(-1, 28 * 28)
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        dout = F.softmax(self.fc3(dout), dim=1)
        return dout


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
