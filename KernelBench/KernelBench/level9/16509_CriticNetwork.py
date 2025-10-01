import torch
import torch.nn.functional as func


class CriticNetwork(torch.nn.Module):

    def __init__(self, s_space, a_space):
        super(CriticNetwork, self).__init__()
        self.s_dense = torch.nn.Linear(s_space, 50)
        self.a_dense = torch.nn.Linear(a_space, 50)
        self.q_dense = torch.nn.Linear(50, 1)

    def forward(self, s, a):
        phi_s = self.s_dense(s)
        phi_a = self.a_dense(a)
        pre_q = func.relu(phi_s + phi_a)
        q_value = self.q_dense(pre_q)
        return q_value


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'s_space': 4, 'a_space': 4}]
