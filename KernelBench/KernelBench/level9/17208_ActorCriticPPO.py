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


class ActorCriticPPO(StochasticContinuousNeuralNet):

    def __init__(self, architecture, weight_init=gauss_weights_init(0, 0.02
        ), activation_functions=None):
        super(ActorCriticPPO, self).__init__()
        if len(architecture) < 2:
            raise Exception(
                'Architecture needs at least two numbers to create network')
        self.activation_functions = activation_functions
        self.layer_list = []
        self.layer_list_val = []
        self.siglog = tor.zeros(1, requires_grad=True)
        self.siglog = nn.Parameter(self.siglog)
        for i in range(len(architecture) - 1):
            self.layer_list.append(nn.Linear(architecture[i], architecture[
                i + 1]))
            setattr(self, 'fc' + str(i), self.layer_list[-1])
        for i in range(len(architecture) - 2):
            self.layer_list_val.append(nn.Linear(architecture[i],
                architecture[i + 1]))
            setattr(self, 'fc_val' + str(i), self.layer_list_val[-1])
        self.layer_list_val.append(nn.Linear(architecture[-2], 1))
        setattr(self, 'fc_val' + str(len(architecture) - 1), self.
            layer_list_val[-1])
        self.apply(weight_init)

    def policy_forward(self, x):
        if self.activation_functions:
            for i, func in enumerate(self.activation_functions):
                x = func(self.layer_list[i](x))
        else:
            for i, layer in enumerate(self.layer_list[:-1]):
                x = self.tanh(layer(x))
        x = self.layer_list[-1](x)
        self._means = self.tanh(x)
        self._dist = Normal(self._means, tor.exp(self.siglog))
        self.sampled = self._dist.rsample()
        x = self.sampled
        return x

    def mu(self):
        return self.means

    def value_forward(self, x):
        if self.activation_functions:
            for i, func in enumerate(self.activation_functions):
                x = func(self.layer_list_val[i](x))
        else:
            for i, layer in enumerate(self.layer_list_val[:-1]):
                x = self.tanh(layer(x))
        x = self.layer_list_val[-1](x)
        return x

    def forward(self, x):
        action = self.policy_forward(x)
        value = self.value_forward(x)
        return tor.cat([action, value], dim=1)

    def __call__(self, state):
        action, value = self.policy_forward(state), self.value_forward(state)
        return action, value

    def sigma(self):
        return self.sigmas

    def mu(self):
        return self._means

    def logprob(self, values):
        return self._dist.log_prob(values)

    def entropy(self):
        return self._dist.entropy()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'architecture': [4, 4]}]
