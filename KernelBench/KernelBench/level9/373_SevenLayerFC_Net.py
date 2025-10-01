import torch
import torch.nn as nn
import torch.nn.functional as F


class SevenLayerFC_Net(nn.Module):

    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate three nn.Linear modules and assign them as
        member variables.
        """
        super(SevenLayerFC_Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, 2048)
        self.linear2 = torch.nn.Linear(2048, 1024)
        self.linear3 = torch.nn.Linear(1024, 512)
        self.linear4 = torch.nn.Linear(512, 256)
        self.linear5 = torch.nn.Linear(256, 128)
        self.linear6 = torch.nn.Linear(128, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = self.linear6(x)
        return F.log_softmax(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4, 'H': 4, 'D_out': 4}]
