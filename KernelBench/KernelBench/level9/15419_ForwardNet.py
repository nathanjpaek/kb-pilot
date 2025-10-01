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


class ForwardNet(torch.nn.Module):
    """
    one hidden layer and one softmax layer.
    Args:
        - input_size:
        - hidden_size:
        - output_size:
        - dropout_p:
    Inputs:
        - x: (seq_len, batch, input_size)
        - x_mask: (batch, seq_len)
    Outputs:
        - beta: (batch, seq_len)
    """

    def __init__(self, input_size, hidden_size, dropout_p):
        super(ForwardNet, self).__init__()
        self.linear_h = torch.nn.Linear(input_size, hidden_size)
        self.linear_o = torch.nn.Linear(hidden_size, 1)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, x, x_mask):
        h = F.relu(self.linear_h(x))
        h = self.dropout(h)
        o = self.linear_o(h)
        o = o.squeeze(2).transpose(0, 1)
        beta = masked_softmax(o, x_mask, dim=1)
        return beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'dropout_p': 0.5}]
