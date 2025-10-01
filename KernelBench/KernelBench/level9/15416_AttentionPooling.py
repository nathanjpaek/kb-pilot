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


class AttentionPooling(torch.nn.Module):
    """
    Attention-Pooling for pointer net init hidden state generate.
    Equal to Self-Attention + MLP
    Modified from r-net.
    Args:
        input_size: The number of expected features in the input uq
        output_size: The number of expected features in the output rq_o

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output
        - **output** (batch, output_size): tensor containing the output features
    """

    def __init__(self, input_size, output_size):
        super(AttentionPooling, self).__init__()
        self.linear_u = torch.nn.Linear(input_size, output_size)
        self.linear_t = torch.nn.Linear(output_size, 1)
        self.linear_o = torch.nn.Linear(input_size, output_size)

    def forward(self, uq, mask):
        q_tanh = F.tanh(self.linear_u(uq))
        q_s = self.linear_t(q_tanh).squeeze(2).transpose(0, 1)
        alpha = masked_softmax(q_s, mask, dim=1)
        rq = torch.bmm(alpha.unsqueeze(1), uq.transpose(0, 1)).squeeze(1)
        rq_o = F.tanh(self.linear_o(rq))
        return rq_o


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
