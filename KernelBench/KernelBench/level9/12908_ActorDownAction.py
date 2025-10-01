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


class ActorDownAction(nn.Module):
    """a top-down module used in bothway message passing that passes messages to children and outputs action"""

    def __init__(self, self_input_dim, action_dim, msg_dim, max_action,
        max_children):
        super(ActorDownAction, self).__init__()
        self.max_action = max_action
        self.action_base = MLPBase(self_input_dim + msg_dim, action_dim)
        self.msg_base = MLPBase(self_input_dim + msg_dim, msg_dim *
            max_children)

    def forward(self, x, m):
        xm = torch.cat((x, m), dim=-1)
        xm = torch.tanh(xm)
        action = self.max_action * torch.tanh(self.action_base(xm))
        msg_down = self.msg_base(xm)
        msg_down = F.normalize(msg_down, dim=-1)
        return action, msg_down


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'self_input_dim': 4, 'action_dim': 4, 'msg_dim': 4,
        'max_action': 4, 'max_children': 4}]
