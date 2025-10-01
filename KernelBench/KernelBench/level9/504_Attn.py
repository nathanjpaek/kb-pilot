import torch
from torch import nn


class Attn(torch.nn.Module):
    """
    Attention:
      feature_dim: dimension of feature embedding
      method: method to calculate attention, (general, dot, concat)
      input_dim: dimension of input embedding, default is the same as feature_dim; method dot is only available when input_dim == feature_dim


    Inputs:
      inputs: batch of inputs; the inp_size is optional; shape=(batch_size, inp_size, feature_dim)
      targets: batch of targets to pay attention; shape=(batch_size, tgt_size, feature_dim)
      mask: optional target binary mask to avoid paying attention to padding item; shape=(batch_size, tgt_size)

    Outputs:
      context: context vector computed as the weighted average of all the encoder outputs; inp_size is optional;shape=(batch_size, inp_size, feature_dim)
      attntion: attention weight paid to the targets; shape=(batch_size, inp_size, tgt_size)
    """

    def __init__(self, feature_dim, method='general', input_dim=None):
        super(Attn, self).__init__()
        self.method = method
        if not input_dim:
            input_dim = feature_dim
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method,
                'is not an appropriate attention method.')
        elif self.method == 'dot' and input_dim != feature_dim:
            raise ValueError(
                'dot does not work when input_dim does not equals to feature_dim'
                )
        if self.method == 'general':
            self.attn = torch.nn.Linear(feature_dim, input_dim)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(input_dim + feature_dim, feature_dim)
            self.v = torch.nn.Parameter(torch.FloatTensor(feature_dim))

    def score(self, inputs, targets):
        if self.method == 'dot':
            return inputs.bmm(targets.transpose(1, 2))
        elif self.method == 'general':
            energy = self.attn(targets)
            return inputs.bmm(energy.transpose(1, 2))
        elif self.method == 'concat':
            inp_size = inputs.size(1)
            tgt_size = targets.size(1)
            inputs_exp = inputs.unsqueeze(2).expand(-1, -1, tgt_size, -1)
            targets_exp = targets.unsqueeze(1).expand(-1, inp_size, -1, -1)
            combined = torch.cat((inputs_exp, targets_exp), 3)
            energy = self.attn(combined).tanh()
            return torch.sum(self.v * energy, dim=3)

    def forward(self, inputs, targets, mask=None):
        inp_shape = inputs.size()
        if len(inp_shape) == 2:
            inputs = inputs.view(inp_shape[0], 1, inp_shape[-1])
        attn_energies = self.score(inputs, targets)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, attn_energies.size(1), -1)
            attn_energies = attn_energies.masked_fill(~mask, float('-inf'))
        attn_weights = nn.functional.softmax(attn_energies, dim=2)
        context = attn_weights.bmm(targets)
        if len(inp_shape) == 2:
            context = context.squeeze(1)
            attn_weights = attn_weights.squeeze(1)
        return context, attn_weights


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_dim': 4}]
