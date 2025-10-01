import torch
from torch import nn
from torch.nn import Parameter


def th(vector):
    return torch.tanh(vector) / 2 + 0.5


def thp(vector):
    return torch.tanh(vector) * 2.2


class Model(nn.Module):
    """
    Base class for models with added support for GradCam activation map
    and a SentiNet defense. The GradCam design is taken from:
https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    If you are not planning to utilize SentiNet defense just import any model
    you like for your tasks.
    """

    def __init__(self):
        super().__init__()
        self.gradient = None

    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def switch_grads(self, enable=True):
        for i, n in self.named_parameters():
            n.requires_grad_(enable)

    def features(self, x):
        """
        Get latent representation, eg logit layer.
        :param x:
        :return:
        """
        raise NotImplementedError

    def forward(self, x, latent=False):
        raise NotImplementedError


class NCModel(Model):

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.pattern = torch.zeros([self.size, self.size], requires_grad=True
            ) + torch.normal(0, 0.5, [self.size, self.size])
        self.mask = torch.zeros([self.size, self.size], requires_grad=True)
        self.mask = Parameter(self.mask)
        self.pattern = Parameter(self.pattern)

    def forward(self, x, latent=None):
        maskh = th(self.mask)
        patternh = thp(self.pattern)
        x = (1 - maskh) * x + maskh * patternh
        return x

    def re_init(self, device):
        p = torch.zeros([self.size, self.size], requires_grad=True
            ) + torch.normal(0, 0.5, [self.size, self.size])
        self.pattern.data = p
        m = torch.zeros([self.size, self.size], requires_grad=True)
        self.mask.data = m


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
