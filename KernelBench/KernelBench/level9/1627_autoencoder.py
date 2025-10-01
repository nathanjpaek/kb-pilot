import torch
import torch.nn.functional as F


class autoencoder(torch.nn.Module):

    def __init__(self, inputDim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        h_size = 128
        c_size = 8
        use_bias = True
        super(autoencoder, self).__init__()
        self.linear1 = torch.nn.Linear(inputDim, h_size, bias=use_bias)
        self.linear2 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear3 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear4 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear5 = torch.nn.Linear(h_size, c_size, bias=use_bias)
        self.linear6 = torch.nn.Linear(c_size, h_size, bias=use_bias)
        self.linear7 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear8 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear9 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear10 = torch.nn.Linear(h_size, inputDim, bias=use_bias)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h = self.linear1(x)
        h = F.relu(h)
        h = self.linear2(h)
        h = F.relu(h)
        h = self.linear3(h)
        h = F.relu(h)
        h = self.linear4(h)
        h = F.relu(h)
        h = self.linear5(h)
        h = F.relu(h)
        h = self.linear6(h)
        h = F.relu(h)
        h = self.linear7(h)
        h = F.relu(h)
        h = self.linear8(h)
        h = F.relu(h)
        h = self.linear9(h)
        h = F.relu(h)
        h = self.linear10(h)
        return h


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inputDim': 4}]
