import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class PolicyGradientLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, nn_outputs, actions, returns):
        output_log_probs = F.log_softmax(nn_outputs, dim=1)
        log_prob_actions_v = returns * output_log_probs[range(len(actions)),
            actions]
        return -log_prob_actions_v.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64),
        torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
