import torch
import torch.utils.data
import torch.nn.functional as F


def masked_softmax(x, m=None, dim=-1):
    """
    Softmax with mask
    :param x:
    :param m:
    :param dim:
    :return:
    """
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-06)
    return softmax


class SelfAttentionGated(torch.nn.Module):
    """
    Self-Attention Gated layer, it`s not weighted sum in the last, but just weighted
    math: \\softmax(W*	anh(W*x)) * x

    Args:
        input_size: The number of expected features in the input x

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output
        - **output** (seq_len, batch, input_size): gated output tensor
    """

    def __init__(self, input_size):
        super(SelfAttentionGated, self).__init__()
        self.linear_g = torch.nn.Linear(input_size, input_size)
        self.linear_t = torch.nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        g_tanh = F.tanh(self.linear_g(x))
        gt = self.linear_t.forward(g_tanh).squeeze(2).transpose(0, 1)
        gt_prop = masked_softmax(gt, x_mask, dim=1)
        gt_prop = gt_prop.transpose(0, 1).unsqueeze(2)
        x_gt = x * gt_prop
        return x_gt


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
