import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd
import torch.nn


class Attn(nn.Module):
    """
    Unit attention operation for alternating co-attention.
    ``https://arxiv.org/pdf/1606.00061.pdf``
    
    .. math::
        \\begin{array}{ll}
        H = \\tanh(W_x * X + (W_g * g)) \\\\
        a = softmax(w_{hx}^T * H}) \\\\
        output = sum a_i * x_i
        \\end{array}

    Args:
        num_hidden: Number of output hidden size
        input_feat_size: Feature size of input image
        guidance_size:  Feature size of attention guidance [default: 0]
        dropout: Dropout rate of attention operation [default: 0.5]

    Inputs:
        - **X** (batch, input_seq_size, input_feat_size): Input image feature
        - **g** (batch, guidance_size): Attention guidance
    """

    def __init__(self, num_hidden, input_feat_size, guidance_size=0,
        dropout=0.5):
        super(Attn, self).__init__()
        self.num_hidden = num_hidden
        self.input_feat_size = input_feat_size
        self.guidance_size = guidance_size
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.W_x = nn.Linear(input_feat_size, num_hidden)
        if guidance_size > 0:
            self.W_g = nn.Linear(guidance_size, num_hidden)
        self.W_hx = nn.Linear(num_hidden, 1)

    def forward(self, X, g=None):
        _batch_size, input_seq_size, input_feat_size = X.size()
        feat = self.W_x(X)
        if g is not None:
            g_emb = self.W_g(g).view(-1, 1, self.num_hidden)
            feat = feat + g_emb.expand_as(feat)
        hidden_feat = torch.tanh(feat)
        if self.dropout is not None:
            hidden_feat = self.dropout(hidden_feat)
        attn_weight = F.softmax(self.W_hx(hidden_feat), dim=1)
        attn_X = torch.bmm(attn_weight.view(-1, 1, input_seq_size), X.view(
            -1, input_seq_size, input_feat_size))
        return attn_X.view(-1, input_feat_size)


class AlterCoAttn(nn.Module):
    """
    Alternatiing Co-Attention mechanism according to ``https://arxiv.org/pdf/1606.00061.pdf``

    .. math::
        \\begin{array}{ll}
            \\hat{s} = Attn(Q, 0) \\\\
            \\hat{v} = Attn(\\hat{s}, V) \\\\
            \\hat{q} = Attn(Q, \\hat{v})
        \\end{array}
    """

    def __init__(self, num_hidden, img_feat_size, ques_feat_size, dropout=0.5):
        super(AlterCoAttn, self).__init__()
        self.num_hidden = num_hidden
        self.img_feat_size = img_feat_size
        self.ques_feat_size = ques_feat_size
        self.attn1 = Attn(num_hidden, ques_feat_size, guidance_size=0,
            dropout=dropout)
        self.attn2 = Attn(num_hidden, img_feat_size, ques_feat_size,
            dropout=dropout)
        self.attn3 = Attn(num_hidden, ques_feat_size, img_feat_size,
            dropout=dropout)

    def forward(self, ques_feat, img_feat):
        """
        : param ques_feat: [batch, ques_feat_size]
        : param img_feat: [batch, img_seq_size, ques_feat_size]
        """
        ques_self_attn = self.attn1(ques_feat.unsqueeze(1), None)
        img_attn_feat = self.attn2(img_feat, ques_self_attn)
        ques_attn_feat = self.attn3(ques_feat.unsqueeze(1), img_attn_feat)
        return ques_attn_feat, img_attn_feat


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_hidden': 4, 'img_feat_size': 4, 'ques_feat_size': 4}]
