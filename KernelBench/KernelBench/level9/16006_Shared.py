from _paritybench_helpers import _mock_config
import torch
import torch.utils.data
import torch.optim
import torch.utils.data.distributed


class Shared(torch.nn.Module):

    def __init__(self, args):
        super(Shared, self).__init__()
        ncha, self.size, _ = args.inputsize
        self.taskcla = args.taskcla
        self.latent_dim = args.latent_dim
        self.nhid = args.units
        self.nlayers = args.nlayers
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(ncha * self.size * self.size, self.nhid)
        if self.nlayers == 3:
            self.fc2 = torch.nn.Linear(self.nhid, self.nhid)
            self.fc3 = torch.nn.Linear(self.nhid, self.latent_dim)
        else:
            self.fc2 = torch.nn.Linear(self.nhid, self.latent_dim)

    def forward(self, x_s):
        h = x_s.view(x_s.size(0), -1)
        h = self.drop(self.relu(self.fc1(h)))
        h = self.drop(self.relu(self.fc2(h)))
        if self.nlayers == 3:
            h = self.drop(self.relu(self.fc3(h)))
        return h


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(inputsize=[4, 4, 4], taskcla=4,
        latent_dim=4, units=4, nlayers=1)}]
