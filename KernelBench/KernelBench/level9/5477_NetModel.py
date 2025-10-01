import torch
import torch.nn.functional as F
import torch.utils.data.dataloader


class NetModel(torch.nn.Module):

    def __init__(self):
        super(NetModel, self).__init__()
        self.hidden = torch.nn.Linear(28 * 28, 300)
        self.output = torch.nn.Linear(300, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)
        return x


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
