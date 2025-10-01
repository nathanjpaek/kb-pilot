import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.distributions import Bernoulli
import torch.distributions


class ReinforcedReceiver(nn.Module):

    def __init__(self, n_bits, n_hidden):
        super(ReinforcedReceiver, self).__init__()
        self.emb_column = nn.Linear(n_bits, n_hidden)
        self.fc1 = nn.Linear(2 * n_hidden, 2 * n_hidden)
        self.fc2 = nn.Linear(2 * n_hidden, n_bits)

    def forward(self, embedded_message, bits):
        embedded_bits = self.emb_column(bits.float())
        x = torch.cat([embedded_bits, embedded_message], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        probs = x.sigmoid()
        distr = Bernoulli(probs=probs)
        entropy = distr.entropy()
        if self.training:
            sample = distr.sample()
        else:
            sample = (probs > 0.5).float()
        log_prob = distr.log_prob(sample).sum(dim=1)
        return sample, log_prob, entropy


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_bits': 4, 'n_hidden': 4}]
