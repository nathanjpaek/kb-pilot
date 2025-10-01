import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Replay:

    def __init__(self):
        self.hidden0 = None
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminal = False
        self.iter = 0
        self.total_reward = 0.0


class A2CAgent:

    def __init__(self):
        self.__iter = 0
        self.last_value = 0.0
        self.replay = Replay()
        self.total_reward = 0.0
        self.deterministic = False

    def decide(self, x):
        self.replay.states.append(x)
        probs, value = self.forward(x)
        distrib = Categorical(probs)
        action = distrib.sample().item(
            ) if not self.deterministic else torch.argmax(probs)
        self.replay.actions.append(action)
        self.last_value = value.item()
        self.__iter += 1
        self.replay.iter = self.__iter
        return action

    def reward(self, r):
        self.replay.rewards.append(r)
        self.total_reward += r

    def forward(self, x):
        raise NotImplementedError

    def on_reset(self, is_terminal):
        raise NotImplementedError

    def set_replay_hidden(self, hidden):
        self.replay.hidden0 = hidden

    def reset(self, new_episode=True):
        if new_episode:
            self.__iter = 0
            self.total_reward = 0.0
        self.replay = Replay()
        self.on_reset(new_episode)

    def end_replay(self, is_terminal):
        replay = self.replay
        replay.is_terminal = is_terminal
        replay.iter = self.__iter
        replay.total_reward = self.total_reward
        if not is_terminal:
            self.reset(new_episode=False)
        if replay.hidden0 is None:
            replay.hidden0 = torch.zeros(1)
        return replay


class Brain(nn.Module, A2CAgent):

    def __init__(self):
        nn.Module.__init__(self)
        A2CAgent.__init__(self)
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        value = self.value_head(x)
        return F.softmax(action_scores, dim=-1), value

    def save(self, filename):
        f = open(filename, 'wb')
        torch.save(self.state_dict(), f)
        f.close()

    def load(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)

    def on_reset(self, new_episode):
        pass


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
