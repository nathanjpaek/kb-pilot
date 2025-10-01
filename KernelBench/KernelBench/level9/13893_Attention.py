import torch
import torch.nn as nn


def masked_softmax(x, m=None, axis=-1):
    """
    Softmax with mask (optional)
    """
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-06)
    return softmax


class TimeDistributedDense(torch.nn.Module):
    """
    input:  x:          batch x time x a
            mask:       batch x time
    output: y:          batch x time x b
    """

    def __init__(self, mlp):
        super(TimeDistributedDense, self).__init__()
        self.mlp = mlp

    def forward(self, x, mask=None):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        y = self.mlp.forward(x)
        y = y.view(x_size[:-1] + (y.size(-1),))
        if mask is not None:
            y = y * mask.unsqueeze(-1)
        return y


class Attention(nn.Module):

    def __init__(self, enc_dim, trg_dim, method='general'):
        super(Attention, self).__init__()
        self.method = method
        if self.method == 'general':
            self.attn = nn.Linear(enc_dim, trg_dim)
        elif self.method == 'concat':
            attn = nn.Linear(enc_dim + trg_dim, trg_dim)
            v = nn.Linear(trg_dim, 1)
            self.attn = TimeDistributedDense(mlp=attn)
            self.v = TimeDistributedDense(mlp=v)
        self.softmax = nn.Softmax()
        if self.method == 'dot':
            self.linear_out = nn.Linear(2 * trg_dim, trg_dim, bias=False)
        else:
            self.linear_out = nn.Linear(enc_dim + trg_dim, trg_dim, bias=False)
        self.tanh = nn.Tanh()

    def score(self, hiddens, encoder_outputs, encoder_mask=None):
        """
        :param hiddens: (batch, trg_len, trg_hidden_dim)
        :param encoder_outputs: (batch, src_len, src_hidden_dim)
        :return: energy score (batch, trg_len, src_len)
        """
        if self.method == 'dot':
            energies = torch.bmm(hiddens, encoder_outputs.transpose(1, 2))
        elif self.method == 'general':
            energies = self.attn(encoder_outputs)
            if encoder_mask is not None:
                energies = energies * encoder_mask.view(encoder_mask.size(0
                    ), encoder_mask.size(1), 1)
            energies = torch.bmm(hiddens, energies.transpose(1, 2))
        elif self.method == 'concat':
            energies = []
            encoder_outputs.size(0)
            src_len = encoder_outputs.size(1)
            for i in range(hiddens.size(1)):
                hidden_i = hiddens[:, i:i + 1, :].expand(-1, src_len, -1)
                concated = torch.cat((hidden_i, encoder_outputs), 2)
                if encoder_mask is not None:
                    concated = concated * encoder_mask.view(encoder_mask.
                        size(0), encoder_mask.size(1), 1)
                energy = self.tanh(self.attn(concated, encoder_mask))
                if encoder_mask is not None:
                    energy = energy * encoder_mask.view(encoder_mask.size(0
                        ), encoder_mask.size(1), 1)
                energy = self.v(energy, encoder_mask).squeeze(-1)
                energies.append(energy)
            energies = torch.stack(energies, dim=1)
            if encoder_mask is not None:
                energies = energies * encoder_mask.view(encoder_mask.size(0
                    ), 1, encoder_mask.size(1))
        return energies.contiguous()

    def forward(self, hidden, encoder_outputs, encoder_mask=None):
        """
        Compute the attention and h_tilde, inputs/outputs must be batch first
        :param hidden: (batch_size, trg_len, trg_hidden_dim)
        :param encoder_outputs: (batch_size, src_len, trg_hidden_dim), if this is dot attention, you have to convert enc_dim to as same as trg_dim first
        :return:
            h_tilde (batch_size, trg_len, trg_hidden_dim)
            attn_weights (batch_size, trg_len, src_len)
            attn_energies  (batch_size, trg_len, src_len): the attention energies before softmax
        """
        """
        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(encoder_outputs.size(0), encoder_outputs.size(1))) # src_seq_len * batch_size
        if torch.cuda.is_available(): attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(encoder_outputs.size(0)):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, transpose to (batch_size * src_seq_len)
        attn = torch.nn.functional.softmax(attn_energies.t())
        # get the weighted context, (batch_size, src_layer_number * src_encoder_dim)
        weighted_context = torch.bmm(encoder_outputs.permute(1, 2, 0), attn.unsqueeze(2)).squeeze(2)  # (batch_size, src_hidden_dim * num_directions)
        """
        batch_size = hidden.size(0)
        src_len = encoder_outputs.size(1)
        trg_len = hidden.size(1)
        context_dim = encoder_outputs.size(2)
        trg_hidden_dim = hidden.size(2)
        attn_energies = self.score(hidden, encoder_outputs)
        if encoder_mask is None:
            attn_weights = torch.nn.functional.softmax(attn_energies.view(-
                1, src_len), dim=1).view(batch_size, trg_len, src_len)
        else:
            attn_energies = attn_energies * encoder_mask.view(encoder_mask.
                size(0), 1, encoder_mask.size(1))
            attn_weights = masked_softmax(attn_energies, encoder_mask.view(
                encoder_mask.size(0), 1, encoder_mask.size(1)), -1)
        weighted_context = torch.bmm(attn_weights, encoder_outputs)
        h_tilde = torch.cat((weighted_context, hidden), 2)
        h_tilde = self.tanh(self.linear_out(h_tilde.view(-1, context_dim +
            trg_hidden_dim)))
        return h_tilde.view(batch_size, trg_len, trg_hidden_dim
            ), attn_weights, attn_energies

    def forward_(self, hidden, context):
        """
        Original forward for DotAttention, it doesn't work if the dim of encoder and decoder are not same
        input and context must be in same dim: return Softmax(hidden.dot([c for c in context]))
        input: batch x hidden_dim
        context: batch x source_len x hidden_dim
        """
        target = self.linear_in(hidden).unsqueeze(2)
        attn = torch.bmm(context, target).squeeze(2)
        attn = self.softmax(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        h_tilde = torch.cat((weighted_context, hidden), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'enc_dim': 4, 'trg_dim': 4}]
