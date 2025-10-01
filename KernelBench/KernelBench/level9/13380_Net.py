import torch
import torch.nn as nn
import torch.nn.functional as F


def set_init(layers):
    for layer in layers:
        nn.init.normal(layer.weight, mean=0.0, std=0.1)
        nn.init.constant(layer.bias, 0.1)


class Net(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 50)
        self.pi2 = nn.Linear(50, 50)
        self.pi3 = nn.Linear(50, a_dim)
        self.v1 = nn.Linear(s_dim, 50)
        self.v2 = nn.Linear(50, 50)
        self.v3 = nn.Linear(50, 1)
        set_init([self.pi1, self.pi2, self.pi3, self.v1, self.v2, self.v3])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = F.relu(self.pi1(x))
        logits = self.pi3(F.relu(self.pi2(pi1)))
        v1 = F.relu(self.v1(x))
        values = self.v3(F.relu(self.v2(v1)))
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'s_dim': 4, 'a_dim': 4}]
