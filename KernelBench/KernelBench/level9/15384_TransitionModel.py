import torch
from torch import nn


def log_clamped(x, eps=0.0001):
    clamped_x = torch.clamp(x, min=eps)
    return torch.log(clamped_x)


def logsumexp(x, dim):
    """
    Differentiable LogSumExp: Does not creates nan gradients when all the inputs are -inf
    Args:
        x : torch.Tensor -  The input tensor
        dim: int - The dimension on which the log sum exp has to be applied
    """
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')
    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))


class TransitionModel(nn.Module):
    """
    Transition Model of the HMM, it represents the probability of transitioning form current state to all other states
    """

    def __init__(self):
        super(TransitionModel, self).__init__()

    def set_staying_and_transitioning_probability(self, staying, transitioning
        ):
        """
        Make reference of the staying and transitioning probabilities as instance parameters of class
        """
        self.staying_probability = staying
        self.transition_probability = transitioning

    def forward(self, log_alpha_scaled, transition_vector, state_lengths):
        """
        It is the product of the past state with transitional probabilities
        and since it is in log scale, the product will be converted to logsumexp
        Args:
            log_alpha_scaled (torch.Tensor): Multiply previous timestep's alphas by transition matrix (in log domain)
                shape: (batch size, N)
            transition_vector (torch.tensor): transition vector for each state
                shape: (N)
            state_lengths (int tensor): Lengths of states in a batch
                shape: (batch)
        """
        T_max = log_alpha_scaled.shape[1]
        transition_probability = torch.sigmoid(transition_vector)
        staying_probability = torch.sigmoid(-transition_vector)
        self.set_staying_and_transitioning_probability(staying_probability,
            transition_probability)
        log_staying_probability = log_clamped(staying_probability)
        log_transition_probability = log_clamped(transition_probability)
        staying = log_alpha_scaled + log_staying_probability
        leaving = log_alpha_scaled + log_transition_probability
        leaving = leaving.roll(1, dims=1)
        leaving[:, 0] = -float('inf')
        mask_tensor = log_alpha_scaled.new_zeros(T_max)
        not_state_lengths_mask = ~(torch.arange(T_max, out=mask_tensor).
            expand(len(state_lengths), T_max) < state_lengths.unsqueeze(1))
        out = logsumexp(torch.stack((staying, leaving), dim=2), dim=2)
        out = out.masked_fill(not_state_lengths_mask, -float('inf'))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
