import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.functional
import torch.cuda


class Highway(nn.Module):
    """The Highway update layer from [srivastava2015]_.

    .. [srivastava2015] Srivastava, R. K., *et al.* (2015).
       `Highway Networks <http://arxiv.org/abs/1505.00387>`_.
       *arXiv*, 1505.00387.
    """

    def __init__(self, input_size: 'int', prev_input_size: 'int'):
        """Instantiate the Highway update layer.

        :param input_size: Current representation size.
        :param prev_input_size: Size of the representation obtained by the previous convolutional layer.
        """
        super().__init__()
        total_size = input_size + prev_input_size
        self.proj = nn.Linear(total_size, input_size)
        self.transform = nn.Linear(total_size, input_size)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, current: 'torch.Tensor', previous: 'torch.Tensor'
        ) ->torch.Tensor:
        """Compute the gated update.

        :param current: Current layer node representations.
        :param previous: Previous layer node representations.
        :returns: The highway-updated inputs.
        """
        concat_inputs = torch.cat((current, previous), 1)
        proj_result = F.relu(self.proj(concat_inputs))
        proj_gate = F.sigmoid(self.transform(concat_inputs))
        gated = proj_gate * proj_result + (1 - proj_gate) * current
        return gated


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'prev_input_size': 4}]
