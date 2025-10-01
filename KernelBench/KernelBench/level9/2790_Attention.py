import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, encoder_dim, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_lin = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.img_lin = nn.Linear(encoder_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        self.concat_lin = nn.Linear(hidden_dim, 1)

    def forward(self, img_features, hidden_state):
        hidden_h = self.hidden_lin(hidden_state).unsqueeze(1)
        img_s = self.img_lin(img_features)
        att_ = self.tanh(img_s + hidden_h)
        e_ = self.concat_lin(att_).squeeze(2)
        alpha = self.softmax(e_)
        context_vec = (img_features * alpha.unsqueeze(2)).sum(1)
        return context_vec, alpha


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'encoder_dim': 4, 'hidden_dim': 4}]
