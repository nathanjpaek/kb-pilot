from torch.nn import Module
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.nn.utils import vector_to_parameters
from torch.nn.utils import parameters_to_vector


def ortho_init(module, nonlinearity=None, weight_scale=1.0, constant_bias=0.0):
    """Applies orthogonal initialization for the parameters of a given module.
    
    Args:
        module (nn.Module): A module to apply orthogonal initialization over its parameters. 
        nonlinearity (str, optional): Nonlinearity followed by forward pass of the module. When nonlinearity
            is not ``None``, the gain will be calculated and :attr:`weight_scale` will be ignored. 
            Default: ``None``
        weight_scale (float, optional): Scaling factor to initialize the weight. Ignored when
            :attr:`nonlinearity` is not ``None``. Default: 1.0
        constant_bias (float, optional): Constant value to initialize the bias. Default: 0.0
        
    .. note::
    
        Currently, the only supported :attr:`module` are elementary neural network layers, e.g.
        nn.Linear, nn.Conv2d, nn.LSTM. The submodules are not supported.
    
    Example::
    
        >>> a = nn.Linear(2, 3)
        >>> ortho_init(a)
    
    """
    if nonlinearity is not None:
        gain = nn.init.calculate_gain(nonlinearity)
    else:
        gain = weight_scale
    if isinstance(module, (nn.RNNBase, nn.RNNCellBase)):
        for name, param in module.named_parameters():
            if 'weight_' in name:
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias_' in name:
                nn.init.constant_(param, constant_bias)
    else:
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, constant_bias)


class MDNHead(Module):

    def __init__(self, in_features, out_features, num_density, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.num_density = num_density
        self.pi_head = nn.Linear(in_features, out_features * num_density)
        ortho_init(self.pi_head, weight_scale=0.01, constant_bias=0.0)
        self.mean_head = nn.Linear(in_features, out_features * num_density)
        ortho_init(self.mean_head, weight_scale=0.01, constant_bias=0.0)
        self.logvar_head = nn.Linear(in_features, out_features * num_density)
        ortho_init(self.logvar_head, weight_scale=0.01, constant_bias=0.0)

    def forward(self, x):
        logit_pi = self.pi_head(x).view(-1, self.num_density, self.out_features
            )
        mean = self.mean_head(x).view(-1, self.num_density, self.out_features)
        logvar = self.logvar_head(x).view(-1, self.num_density, self.
            out_features)
        std = torch.exp(0.5 * logvar)
        return logit_pi, mean, std

    def loss(self, logit_pi, mean, std, target):
        """Calculate the MDN loss function. 
        
        The loss function (negative log-likelihood) is defined by:
        
        .. math::
            L = -\\frac{1}{N}\\sum_{n=1}^{N}\\ln \\left( \\sum_{k=1}^{K}\\prod_{d=1}^{D} \\pi_{k}(x_{n, d})
            \\mathcal{N}\\left( \\mu_k(x_{n, d}), \\sigma_k(x_{n,d}) \\right) \\right)
            
        For better numerical stability, we could use log-scale:
        
        .. math::
            L = -\\frac{1}{N}\\sum_{n=1}^{N}\\ln \\left( \\sum_{k=1}^{K}\\exp \\left\\{ \\sum_{d=1}^{D} 
            \\ln\\pi_{k}(x_{n, d}) + \\ln\\mathcal{N}\\left( \\mu_k(x_{n, d}), \\sigma_k(x_{n,d}) 
            \\right) \\right\\} \\right) 
        
        .. note::
        
            One should always use the second formula via log-sum-exp trick. The first formula
            is numerically unstable resulting in +/- ``Inf`` and ``NaN`` error. 
        
        The log-sum-exp trick is defined by
        
        .. math::
            \\log\\sum_{i=1}^{N}\\exp(x_i) = a + \\log\\sum_{i=1}^{N}\\exp(x_i - a)
            
        where :math:`a = \\max_i(x_i)`
        
        Args:
            logit_pi (Tensor): the logit of mixing coefficients, shape [N, K, D]
            mean (Tensor): mean of Gaussian mixtures, shape [N, K, D]
            std (Tensor): standard deviation of Gaussian mixtures, shape [N, K, D]
            target (Tensor): target tensor, shape [N, D]

        Returns:
            Tensor: calculated loss
        """
        target = target.unsqueeze(1)
        log_pi = F.log_softmax(logit_pi, dim=1)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(target)
        joint_log_probs = torch.sum(log_pi + log_probs, dim=-1, keepdim=False)
        loss = torch.logsumexp(joint_log_probs, dim=-1, keepdim=False)
        loss = -loss.mean(0)
        return loss

    def sample(self, logit_pi, mean, std, tau=1.0):
        """Sample from Gaussian mixtures using reparameterization trick.
        
        - Firstly sample categorically over mixing coefficients to determine a specific Gaussian
        - Then sample from selected Gaussian distribution
        
        Args:
            logit_pi (Tensor): the logit of mixing coefficients, shape [N, K, D]
            mean (Tensor): mean of Gaussian mixtures, shape [N, K, D]
            std (Tensor): standard deviation of Gaussian mixtures, shape [N, K, D]
            tau (float): temperature during sampling, it controls uncertainty. 
                * If :math:`\\tau > 1`: increase uncertainty
                * If :math:`\\tau < 1`: decrease uncertainty
        
        Returns:
            Tensor: sampled data with shape [N, D]
        """
        N, K, D = logit_pi.shape
        pi = F.softmax(logit_pi / tau, dim=1)
        pi = pi.permute(0, 2, 1).view(-1, K)
        mean = mean.permute(0, 2, 1).view(-1, K)
        std = std.permute(0, 2, 1).view(-1, K)
        pi_samples = Categorical(pi).sample()
        mean = mean[torch.arange(N * D), pi_samples]
        std = std[torch.arange(N * D), pi_samples]
        eps = torch.randn_like(std)
        samples = mean + eps * std * np.sqrt(tau)
        samples = samples.view(N, D)
        return samples


class Module(nn.Module):
    """Wrap PyTorch nn.module to provide more helper functions. """

    def __init__(self, **kwargs):
        super().__init__()
        for key, val in kwargs.items():
            self.__setattr__(key, val)

    @property
    def num_params(self):
        """Returns the total number of parameters in the neural network. """
        return sum(param.numel() for param in self.parameters())

    @property
    def num_trainable_params(self):
        """Returns the total number of trainable parameters in the neural network."""
        return sum(param.numel() for param in self.parameters() if param.
            requires_grad)

    @property
    def num_untrainable_params(self):
        """Returns the total number of untrainable parameters in the neural network. """
        return sum(param.numel() for param in self.parameters() if not
            param.requires_grad)

    def to_vec(self):
        """Returns the network parameters as a single flattened vector. """
        return parameters_to_vector(parameters=self.parameters())

    def from_vec(self, x):
        """Set the network parameters from a single flattened vector.
        
        Args:
            x (Tensor): A single flattened vector of the network parameters with consistent size.
        """
        vector_to_parameters(vec=x, parameters=self.parameters())

    def save(self, f):
        """Save the network parameters to a file. 
        
        It complies with the `recommended approach for saving a model in PyTorch documentation`_. 
        
        .. note::
            It uses the highest pickle protocol to serialize the network parameters. 
        
        Args:
            f (str): file path. 
            
        .. _recommended approach for saving a model in PyTorch documentation:
            https://pytorch.org/docs/master/notes/serialization.html#best-practices
        """
        torch.save(obj=self.state_dict(), f=f, pickle_protocol=pickle.
            HIGHEST_PROTOCOL)

    def load(self, f):
        """Load the network parameters from a file. 
        
        It complies with the `recommended approach for saving a model in PyTorch documentation`_. 
        
        Args:
            f (str): file path. 
            
        .. _recommended approach for saving a model in PyTorch documentation:
            https://pytorch.org/docs/master/notes/serialization.html#best-practices
        """
        self.load_state_dict(torch.load(f))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'num_density': 4}]
