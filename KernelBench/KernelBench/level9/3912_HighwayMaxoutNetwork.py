import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.

    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e+30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)
    return probs


class HighwayMaxoutNetwork(nn.Module):
    """HMN network for dynamic decoder.

    Based on the Co-attention paper:

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """

    def __init__(self, mod_out_size, hidden_size, max_out_pool_size):
        super(HighwayMaxoutNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.maxout_pool_size = max_out_pool_size
        None
        self.r = nn.Linear(2 * mod_out_size + hidden_size, hidden_size,
            bias=False)
        self.W1 = nn.Linear(mod_out_size + hidden_size, max_out_pool_size *
            hidden_size)
        self.W2 = nn.Linear(hidden_size, max_out_pool_size * hidden_size)
        self.W3 = nn.Linear(2 * hidden_size, max_out_pool_size)

    def forward(self, mod, h_i, u_s_prev, u_e_prev, mask):
        batch_size, seq_len, _mod_out_size = mod.shape
        r = F.tanh(self.r(torch.cat((h_i, u_s_prev, u_e_prev), 1)))
        r_expanded = r.unsqueeze(1).expand(batch_size, seq_len, self.
            hidden_size).contiguous()
        W1_inp = torch.cat((mod, r_expanded), 2)
        m_t_1 = self.W1(W1_inp)
        m_t_1 = m_t_1.view(batch_size, seq_len, self.maxout_pool_size, self
            .hidden_size)
        m_t_1, _ = m_t_1.max(2)
        assert m_t_1.shape == (batch_size, seq_len, self.hidden_size)
        m_t_2 = self.W2(m_t_1)
        m_t_2 = m_t_2.view(batch_size, seq_len, self.maxout_pool_size, self
            .hidden_size)
        m_t_2, _ = m_t_2.max(2)
        alpha_in = torch.cat((m_t_1, m_t_2), 2)
        alpha = self.W3(alpha_in)
        logits, _ = alpha.max(2)
        log_p = masked_softmax(logits, mask, log_softmax=True)
        return log_p


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'mod_out_size': 4, 'hidden_size': 4, 'max_out_pool_size': 4}]
