import torch
import numpy as np
from torch.autograd import Variable


class Net(torch.nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super(Net, self).__init__()
        self.w1 = torch.nn.Linear(n_in, n_hidden)
        self.w2 = torch.nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = torch.tanh(self.w1(x))
        x = self.w2(x)
        return x

    def my_train(self, xtrain, ytrain, num_epochs):
        """
        Train the network

        Parameters
        ----------

        xtrain : np.ndarray
            Inputs

        ytrain : np.ndarray
            Corresponding desired outputs
        """
        xtrain = Variable(torch.FloatTensor(xtrain))
        ytrain = Variable(torch.FloatTensor(ytrain))
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-05)
        for t in range(num_epochs):
            optimizer.zero_grad()
            y_pred = self(xtrain)
            loss = criterion(y_pred, ytrain)
            loss.backward()
            optimizer.step()
        None

    def call_numpy(self, x: 'np.ndarray'):
        """
        Call the network with numpy input and output
        """
        x_tensor = Variable(torch.FloatTensor(x))
        out = self(x_tensor)
        return out.detach().numpy()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_hidden': 4, 'n_out': 4}]
