import torch
import torch.nn as nn


class ActionScoring(nn.Module):
    """ Linearly mapping h and v to the same dimension, 
        and do a elementwise multiplication and a linear scoring. """

    def __init__(self, action_size, hidden_size, dot_size: 'int'=256):
        super(ActionScoring, self).__init__()
        self.linear_act = nn.Linear(action_size, dot_size, bias=True)
        self.linear_hid = nn.Linear(hidden_size, dot_size, bias=True)
        self.linear_out = nn.Linear(dot_size, 1, bias=True)

    def forward(self, act_cands, h_tilde):
        """ Compute logits of action candidates
        :param act_cands: torch.Tensor(batch, num_candidates, action_emb_size)
        :param h_tilde: torch.Tensor(batch, hidden_size)
        
        Return -> torch.Tensor(batch, num_candidates)
        """
        target = self.linear_hid(h_tilde).unsqueeze(1)
        context = self.linear_act(act_cands)
        product = torch.mul(context, target)
        logits = self.linear_out(product).squeeze(2)
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'action_size': 4, 'hidden_size': 4}]
