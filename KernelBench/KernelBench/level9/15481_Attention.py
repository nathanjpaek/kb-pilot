import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, dim, dim_embed):
        super(Encoder, self).__init__()
        self.embed = nn.Conv1d(dim, dim_embed, 1)
        return

    def forward(self, input):
        input_2 = input.permute(0, 2, 1)
        out = self.embed(input_2)
        return out.permute(0, 2, 1)


class Attention(nn.Module):

    def __init__(self, dim_embed, embeding_type='conv1d', tanh_exp=0):
        super(Attention, self).__init__()
        self.dim_embed = dim_embed
        if embeding_type == 'conv1d':
            self.proj = Encoder(dim_embed, dim_embed)
            self.w_a = Encoder(dim_embed * 3, dim_embed)
            self.v_a = nn.Parameter(torch.randn(dim_embed))
        else:
            self.proj = nn.Linear(dim_embed, dim_embed)
            self.w_a = nn.Linear(dim_embed * 3, dim_embed)
            self.v_a = nn.Parameter(torch.randn(dim_embed))
        self.tanh_exp = tanh_exp
        return

    def forward(self, encoded_static, encoded_dynamic, decoder_output):
        n_nodes = encoded_static.shape[1]
        x_t = torch.cat((encoded_static, encoded_dynamic), dim=2)
        proj_dec = self.proj(decoder_output.unsqueeze(1)).repeat(1, n_nodes, 1)
        hidden = torch.cat((x_t, proj_dec), dim=2)
        u_t = torch.matmul(self.v_a, torch.tanh(self.w_a(hidden)).permute(0,
            2, 1))
        if self.tanh_exp > 0:
            logits = self.tanh_exp * torch.tanh(u_t)
        else:
            logits = u_t
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim_embed': 4}]
