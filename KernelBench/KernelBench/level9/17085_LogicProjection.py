import torch
from torch import nn
import torch.nn.functional as F


class LogicProjection(nn.Module):

    def __init__(self, entity_dim, relation_dim, hidden_dim, num_layers,
        bounded):
        super(LogicProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bounded = bounded
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.
            hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)
        for nl in range(2, num_layers + 1):
            setattr(self, 'layer{}'.format(nl), nn.Linear(self.hidden_dim,
                self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, 'layer{}'.format(nl)).weight)

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, 'layer{}'.format(nl))(x))
        x = self.layer0(x)
        x = torch.sigmoid(x)
        if self.bounded:
            lower, upper = torch.chunk(x, 2, dim=-1)
            upper = lower + upper * (1 - lower)
            x = torch.cat([lower, upper], dim=-1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'entity_dim': 4, 'relation_dim': 4, 'hidden_dim': 4,
        'num_layers': 1, 'bounded': 4}]
