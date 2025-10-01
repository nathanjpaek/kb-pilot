import torch
import numpy as np
import torch as tor
from torch import nn


class SaveableModel(object):

    def save(self, path):
        tor.save(self, path)

    @classmethod
    def load(cls, path):
        return tor.load(path)

    @classmethod
    def load_best(cls, path):
        assert os.path.isdir(path)
        best_models = glob.glob(os.path.join(path, '*best*'))
        assert not len(best_models > 1)
        return tor.load(os.path.join(path, best_models[0]))


class NeuralNet(nn.Module, SaveableModel):

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def action(self, x):
        pass


class StochasticNeuralNet(NeuralNet):

    def __init__(self):
        super(StochasticNeuralNet, self).__init__()

    def sample_action(self, action_distribution=None):
        if not action_distribution:
            action_distribution = self.out
        action_distribution = action_distribution.cpu().data.numpy()
        action = np.random.choice(action_distribution.squeeze(), p=
            action_distribution.squeeze())
        action = np.argmax(action_distribution == action)
        return action


class PolicyAHG(StochasticNeuralNet):

    def __init__(self, input_size, output_size):
        super(PolicyAHG, self).__init__()
        self.f1 = nn.Linear(input_size, 32)
        self.f2 = nn.Linear(32, output_size)

    def forward(self, x):
        out = self.f1(x)
        out = self.tanh(out)
        out = self.f2(out)
        out = self.softmax(out)
        self.out = out
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
