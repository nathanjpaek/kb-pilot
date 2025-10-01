import torch
from torch import nn
import torch.distributions as tdist


class EmissionModel(nn.Module):
    """
    Emission Model of the HMM, it represents the probability of emitting an observation based on the current state
    """

    def __init__(self):
        super(EmissionModel, self).__init__()
        self.distribution_function = tdist.normal.Normal

    def sample(self, means, stds):
        """
        Draws a Sample from each distribution
        """
        return self.distribution_function(means, stds).sample()

    def forward(self, x_t, means, stds, state_lengths):
        """
        Calculates the log probability of the the given data (x_t) being observed from states
        Args:
             x_t (float tensor) : observation at current time step
                shape: (batch, feature_dim)
             means (float tensor): means of the distributions of hidden states
                shape: (batch, hidden_state, feature_dim)
             stds (float tensor): standard deviations of the distributions of the hidden states
                shape: (feature_dim)     tdist.normal.Normal will broadcast to the shape needed
            state_lengths (int tensor): Lengths of states in a batch
                shape: (batch)

        Returns:
            out (float tensor): observation log likelihoods, expressing the probability of an observation
                being generated from a state i
                shape: (batch, hidden_state)
        """
        T_max = means.shape[1]
        emission_dists = self.distribution_function(means, stds)
        x_t = x_t.unsqueeze(1)
        out = emission_dists.log_prob(x_t)
        mask_tensor = x_t.new_zeros(T_max)
        state_lengths_mask = (torch.arange(T_max, out=mask_tensor).expand(
            len(state_lengths), T_max) < state_lengths.unsqueeze(1)).unsqueeze(
            2)
        out = torch.sum(out * state_lengths_mask, dim=2)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
