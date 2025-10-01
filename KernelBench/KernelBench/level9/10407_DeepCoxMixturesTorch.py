import torch
import torch.nn as nn


def create_representation(inputdim, layers, activation):
    """Helper function to generate the representation function for DSM.

  Deep Survival Machines learns a representation (\\ Phi(X) \\) for the input
  data. This representation is parameterized using a Non Linear Multilayer
  Perceptron (`torch.nn.Module`). This is a helper function designed to
  instantiate the representation for Deep Survival Machines.

  .. warning::
    Not designed to be used directly.

  Parameters
  ----------
  inputdim: int
      Dimensionality of the input features.
  layers: list
      A list consisting of the number of neurons in each hidden layer.
  activation: str
      Choice of activation function: One of 'ReLU6', 'ReLU' or 'SeLU'.

  Returns
  ----------
  an MLP with torch.nn.Module with the specfied structure.

  """
    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()
    modules = []
    prevdim = inputdim
    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias=False))
        modules.append(act)
        prevdim = hidden
    return nn.Sequential(*modules)


class DeepCoxMixturesTorch(nn.Module):
    """PyTorch model definition of the Deep Cox Mixture Survival Model.

  The Cox Mixture involves the assumption that the survival function
  of the individual to be a mixture of K Cox Models. Conditioned on each
  subgroup Z=k; the PH assumptions are assumed to hold and the baseline
  hazard rates is determined non-parametrically using an spline-interpolated
  Breslow's estimator.
  """

    def _init_dcm_layers(self, lastdim):
        self.gate = torch.nn.Linear(lastdim, self.k, bias=False)
        self.expert = torch.nn.Linear(lastdim, self.k, bias=False)

    def __init__(self, inputdim, k, gamma=1, use_activation=False, layers=
        None, optimizer='Adam'):
        super(DeepCoxMixturesTorch, self).__init__()
        if not isinstance(k, int):
            raise ValueError(f'k must be int, but supplied k is {type(k)}')
        self.k = k
        self.optimizer = optimizer
        if layers is None:
            layers = []
        self.layers = layers
        if len(layers) == 0:
            lastdim = inputdim
        else:
            lastdim = layers[-1]
        self._init_dcm_layers(lastdim)
        self.embedding = create_representation(inputdim, layers, 'ReLU6')
        self.gamma = gamma
        self.use_activation = use_activation

    def forward(self, x):
        gamma = self.gamma
        x = self.embedding(x)
        if self.use_activation:
            log_hazard_ratios = gamma * torch.nn.Tanh()(self.expert(x))
        else:
            log_hazard_ratios = torch.clamp(self.expert(x), min=-gamma, max
                =gamma)
        log_gate_prob = torch.nn.LogSoftmax(dim=1)(self.gate(x))
        return log_gate_prob, log_hazard_ratios


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inputdim': 4, 'k': 4}]
