import torch
import torch.nn as nn
import torch.nn.functional as F


class dnn(nn.Module):

    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def __init__(self, G_in, G_out, w1, w2, w3):
        super(dnn, self).__init__()
        self.fc1 = nn.Linear(G_in, w1)
        self.fc2 = nn.Linear(w1, w2)
        self.fc3 = nn.Linear(w2, w3)
        self.out = nn.Linear(w3, G_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.out(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'G_in': 4, 'G_out': 4, 'w1': 4, 'w2': 4, 'w3': 4}]
