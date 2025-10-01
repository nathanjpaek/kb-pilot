import torch
import torch as tor
from torch import nn
from torch.distributions import Normal


def gauss_weights_init(mu, std):

    def init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(mu, std)
    return init


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


class StochasticContinuousNeuralNet(NeuralNet):

    def __init__(self):
        super(StochasticContinuousNeuralNet, self).__init__()

    def sigma(self):
        pass

    def mu(self):
        pass


class GaussianPolicy(StochasticContinuousNeuralNet):

    def __init__(self, architecture, weight_init=gauss_weights_init(0, 0.02
        ), activation_functions=None):
        super(GaussianPolicy, self).__init__()
        if len(architecture) < 2:
            raise Exception(
                'Architecture needs at least two numbers to create network')
        self.activation_functions = activation_functions
        self.layer_list = []
        for i in range(len(architecture) - 1):
            self.layer_list.append(nn.Linear(architecture[i], architecture[
                i + 1]))
            setattr(self, 'fc' + str(i), self.layer_list[-1])
        self.apply(weight_init)

    def forward(self, x):
        if self.activation_functions:
            for i, func in enumerate(self.activation_functions):
                x = func(self.layer_list[i](x))
        else:
            for i, layer in enumerate(self.layer_list[:-1]):
                x = self.relu(layer(x))
        x = self.layer_list[-1](x)
        self.means = self.tanh(x[None, :int(x.shape[1] / 2)])
        self.sigmas = self.softmax(x[None, int(x.shape[1] / 2):])
        self.dist = Normal(self.means, self.sigmas)
        self.sampled = self.dist.rsample()
        x = self.sampled
        self.out = x
        return x

    def sigma(self):
        return self.sigmas

    def mu(self):
        return self.means

    def log_prob(self, values):
        return self.dist.log_prob(values)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'architecture': [4, 4]}]
