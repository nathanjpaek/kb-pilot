import torch
import torch.nn as nn
import torch.nn.functional as F


def set_init(layers):
    for layer in layers:
        nn.init.normal(layer.weight, mean=0.0, std=0.3)
        nn.init.constant(layer.bias, 0.3)


class Net(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.b1 = nn.Linear(s_dim, 32)
        self.b2 = nn.Linear(32, 24)
        self.pi = nn.Linear(24, a_dim)
        self.v = nn.Linear(24, 1)
        set_init([self.b1, self.b2, self.pi, self.v])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        b1_o = F.leaky_relu(self.b1(x))
        b2_o = F.leaky_relu(self.b2(b1_o))
        logits = self.pi(b2_o)
        values = self.v(b2_o)
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
        return a_loss.mean(), c_loss.mean(), total_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'s_dim': 4, 'a_dim': 4}]
