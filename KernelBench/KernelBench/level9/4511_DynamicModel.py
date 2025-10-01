import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):

    def forward(self, x):
        if len(x.size()) > 1:
            return x / x.norm(p=2, dim=1, keepdim=True)
        else:
            return x / x.norm(p=2)


class NonLinearModel(nn.Module):

    def __init__(self, inputs, outputs, hiddens=32):
        super(NonLinearModel, self).__init__()
        self.l1 = nn.Linear(inputs, hiddens)
        self.l2 = nn.Linear(hiddens, outputs)
        self.norm = L2Norm()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = self.norm(x)
        return x


class LinearModel(nn.Module):

    def __init__(self, inputs, outputs):
        super(LinearModel, self).__init__()
        self.l1 = nn.Linear(inputs, outputs)

    def forward(self, x):
        return self.l1(x)


class DynamicModel(nn.Module):

    def __init__(self, encode_size, action_size):
        super(DynamicModel, self).__init__()
        self.encode_size = encode_size
        self.action_size = action_size
        self.action_transition_mapping = NonLinearModel(self.action_size,
            self.encode_size)
        self.action_reward_mapping = NonLinearModel(self.action_size, self.
            encode_size)
        self.transition = LinearModel(self.encode_size, self.encode_size)
        self.reward = LinearModel(self.encode_size, 1)
        self.norm = L2Norm()

    def forward(self, feature, action):
        transition_action = self.action_transition_mapping(action)
        reward_action = self.action_reward_mapping(action)
        predict_next = self.transition(torch.multiply(feature,
            transition_action))
        predict_reward = self.reward(torch.multiply(feature, reward_action))
        predict_next = self.norm(predict_next)
        return predict_next, predict_reward


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'encode_size': 4, 'action_size': 4}]
