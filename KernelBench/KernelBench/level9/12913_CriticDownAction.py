import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBase(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(MLPBase, self).__init__()
        self.l1 = nn.Linear(num_inputs, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, num_outputs)

    def forward(self, inputs):
        x = F.relu(self.l1(inputs))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class CriticDownAction(nn.Module):
    """a top-down module used in bothway message passing that passes messages to children and outputs q-values"""

    def __init__(self, self_input_dim, action_dim, msg_dim, max_children):
        super(CriticDownAction, self).__init__()
        self.baseQ1 = MLPBase(self_input_dim + action_dim + msg_dim, 1)
        self.baseQ2 = MLPBase(self_input_dim + action_dim + msg_dim, 1)
        self.msg_base = MLPBase(self_input_dim + msg_dim, msg_dim *
            max_children)

    def forward(self, x, u, m):
        xum = torch.cat([x, u, m], dim=-1)
        x1 = self.baseQ1(xum)
        x2 = self.baseQ2(xum)
        xm = torch.cat([x, m], dim=-1)
        xm = torch.tanh(xm)
        msg_down = self.msg_base(xm)
        msg_down = F.normalize(msg_down, dim=-1)
        return x1, x2, msg_down

    def Q1(self, x, u, m):
        xum = torch.cat([x, u, m], dim=-1)
        x1 = self.baseQ1(xum)
        xm = torch.cat([x, m], dim=-1)
        xm = torch.tanh(xm)
        msg_down = self.msg_base(xm)
        msg_down = F.normalize(msg_down, dim=-1)
        return x1, msg_down


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'self_input_dim': 4, 'action_dim': 4, 'msg_dim': 4,
        'max_children': 4}]
