import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.modules.loss import _Loss
import torch.jit


class BinaryNLLEntropy(_Loss):

    def __init__(self, size_average=True):
        super(BinaryNLLEntropy, self).__init__()
        self.size_average = size_average

    def forward(self, net_output, label_output):
        """
        :param net_output: batch_size x
        :param labels:
        :return:
        """
        batch_size = net_output.size(0)
        loss = F.binary_cross_entropy_with_logits(net_output, label_output,
            size_average=self.size_average)
        if self.size_average is False:
            loss /= batch_size
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
