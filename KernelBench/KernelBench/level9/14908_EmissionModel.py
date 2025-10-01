import torch
import torch.utils.data


class EmissionModel(torch.nn.Module):
    """
	- forward(): computes the log probability of an observation.
	- sample(): given a state, sample an observation for that state.
	"""

    def __init__(self, N, M):
        super(EmissionModel, self).__init__()
        self.N = N
        self.M = M
        self.unnormalized_emission_matrix = torch.nn.Parameter(torch.randn(
            N, M))

    def forward(self, x_t):
        """
		x_t : LongTensor of shape (batch size)

		Get observation probabilities
		"""
        emission_matrix = torch.nn.functional.log_softmax(self.
            unnormalized_emission_matrix, dim=1)
        out = emission_matrix[:, x_t].transpose(0, 1)
        return out


def get_inputs():
    return [torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'N': 4, 'M': 4}]
