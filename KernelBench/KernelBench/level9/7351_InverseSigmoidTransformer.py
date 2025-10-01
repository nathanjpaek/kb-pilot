import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.distributions.utils import probs_to_logits


class Bijection(nn.Module):
    """
    An invertible transformation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, context):
        """
        :param inputs: [..., D]
        :param context: [..., H] or None
        :return: [..., D] outputs and [..., D] log determinant of Jacobian
        """
        raise NotImplementedError('Implement me!')

    def inverse(self, outputs, context):
        """
        :param outputs: [..., D]
        :param context: [..., H] or None
        :return: [..., D] inputs and [..., D] log determinant of Jacobian of inverse transform
        """
        raise NotImplementedError('Implement me!')


class InverseSigmoidTransformer(Bijection):
    """
    Maps inputs from R to (0, 1) using a sigmoid.
    """

    def __init__(self):
        super().__init__()

    def forward(self, outputs, context=None):
        inputs = probs_to_logits(outputs, is_binary=True)
        log_p, log_q = -F.softplus(-inputs), -F.softplus(inputs)
        return inputs, -log_p - log_q

    def inverse(self, inputs, context=None):
        log_p, log_q = -F.softplus(-inputs), -F.softplus(inputs)
        outputs = torch.sigmoid(inputs)
        return outputs, log_p + log_q


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
