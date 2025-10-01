import torch
import torch.nn.functional as func


class ActorNetwork(torch.nn.Module):

    def __init__(self, s_space, a_space):
        super(ActorNetwork, self).__init__()
        self.first_dense = torch.nn.Linear(s_space, 50)
        self.second_dense = torch.nn.Linear(50, a_space)

    def forward(self, s):
        phi_s = func.relu(self.first_dense(s))
        prb_a = func.sigmoid(self.second_dense(phi_s))
        return prb_a


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'s_space': 4, 'a_space': 4}]
