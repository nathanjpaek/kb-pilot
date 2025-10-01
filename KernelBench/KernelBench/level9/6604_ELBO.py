import torch
import torch.nn.functional
import torch.nn as nn


class ELBO(nn.Module):

    def __init__(self, train_size, loss_function=nn.MSELoss()):
        """
        Quantify the Evidence Lower Bound (ELBO) and provide the total loss.
        """
        super(ELBO, self).__init__()
        self.train_size = train_size
        self.loss_function = loss_function

    def forward(self, input, target, kl, kl_weight=1.0):
        """
        Kullback-Leibler divergence. This comes from
        the euqation (4) in Shridhar et. al. 2019, which consists of likelihood cost
        (is dependent on data) and complexity cost (id dependent on distribution).
        """
        assert not target.requires_grad
        likelihood_cost = self.loss_function(input, target)
        complexity_cost = kl_weight * kl
        total_loss = likelihood_cost + complexity_cost
        return total_loss, likelihood_cost, complexity_cost


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'train_size': False}]
