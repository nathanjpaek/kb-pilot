import torch
import torch.nn as nn


class Selection(nn.Module):
    """
    Selection neurons to sample from a latent representation for a decoder agent.
    An abstract representation :math:`l_i` is disturbed by a value :math:`r_i` sampled from a normal 
    standard distribution which is scaled by the selection neuron :math:`s_i`.

    ..math::
        n_i \\sim l_i + \\sigma_{l_i} 	imes \\exp(s_i) 	imes r_i

    where :math:`\\sigma_{l_i}` is the standard deviation over the batch. 
    If the selection neuron has a low (i.e. negative) value, the latent variable is passed to the agent. 
    If the selection neuron has a high value (i.e. close to zero), the latent variable is rendered useless to the agent.

    Args:
        num_selectors (int): Number of selection neurons, i.e. latent variables.

        **kwargs:
            init_selectors (float): Initial value for selection neurons. Default: -10.
    """

    def __init__(self, num_selectors, init_selectors=-10.0):
        super(Selection, self).__init__()
        select = torch.Tensor([init_selectors for _ in range(num_selectors)])
        self.selectors = nn.Parameter(select)

    def forward(self, x, rand, std_dev=None):
        """
        The forward pass for the selection neurons.

        Args:
            x (torch.Tensor): The input array of shape (batch_size, size_latent).
            rand (torch.Tensor): Random samples from standard normal distribution of size (batch_size, size_latent).

            **kwargs:
                std_dev (:class:`torch.Tensor` or :class:`NoneType`): The standard deviation calculated throughout 
                                                                      episodes. Needs to be specified for prediction. 
                                                                      Default: None.
        
        Returns:
            sample (torch.Tensor): Sample from a distribution around latent variables.
        """
        selectors = self.selectors.expand_as(x)
        if std_dev is None:
            std = x.std(dim=0).expand_as(x)
        else:
            std = std_dev
        sample = x + std * torch.exp(selectors) * rand
        return sample


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_selectors': 4}]
