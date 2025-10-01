import torch
import torch.nn as nn


class InstrShifting(nn.Module):
    """ Sub-Instruction Shifting Module.
        Decide whether the current subinstruction will 
        be completed by the next action or not. """

    def __init__(self, rnn_hidden_size, shift_hidden_size, action_emb_size,
        max_subinstr_size, drop_ratio):
        super(InstrShifting, self).__init__()
        self.drop = nn.Dropout(p=drop_ratio)
        self.linear0 = nn.Linear(rnn_hidden_size, shift_hidden_size, bias=False
            )
        self.linear1 = nn.Linear(rnn_hidden_size + shift_hidden_size +
            action_emb_size, shift_hidden_size, bias=False)
        self.linear2 = nn.Linear(max_subinstr_size, shift_hidden_size, bias
            =False)
        self.linear3 = nn.Linear(2 * shift_hidden_size, 1, bias=False)

    def forward(self, h_t, m_t, a_t_cur, weighted_ctx, e_t):
        """ Propogate through the network. 
        :param h_t:          torch.Tensor, batch x rnn_hidden_size
        :param m_t:          torch.Tensor, batch x rnn_hidden_size
        :param a_t_cur:      torch.Tensor, batch x action_emb_size
        :param weighted_ctx: torch.Tensor, batch x rnn_hidden_size
        :param e_t:          torch.Tensor, batch x max_subinstr_size
        """
        proj_h = self.linear0(self.drop(h_t))
        concat_input = torch.cat((proj_h, a_t_cur, weighted_ctx), 1)
        h_t_c = torch.sigmoid(self.linear1(concat_input)) * torch.tanh(m_t)
        proj_e = self.linear2(e_t)
        concat_input = torch.cat((proj_e, self.drop(h_t_c)), 1)
        p_t_s = torch.sigmoid(self.linear3(concat_input))
        return p_t_s.squeeze()


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'rnn_hidden_size': 4, 'shift_hidden_size': 4,
        'action_emb_size': 4, 'max_subinstr_size': 4, 'drop_ratio': 0.5}]
