import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, dim, dim_embed):
        super(Encoder, self).__init__()
        self.embed = nn.Conv1d(dim, dim_embed, 1)
        return

    def forward(self, input):
        input_2 = input.permute(0, 2, 1)
        out = self.embed(input_2)
        return out.permute(0, 2, 1)


class Critic(nn.Module):

    def __init__(self, batch_size, n_nodes, dim_s, dim_embed, embeding_type
        ='conv1d'):
        super(Critic, self).__init__()
        self.dim_embed = dim_embed
        if embeding_type == 'conv1d':
            self.project_s = Encoder(dim_s, dim_embed)
            self.w_a = Encoder(dim_embed, dim_embed)
            self.w_c = Encoder(dim_embed * 2, dim_embed)
            self.v_a = nn.Parameter(torch.randn(dim_embed))
            self.v_c = nn.Parameter(torch.randn(dim_embed))
        else:
            self.project_s = nn.Linear(dim_s, dim_embed)
            self.w_a = nn.Linear(dim_embed, dim_embed)
            self.w_c = nn.Linear(dim_embed * 2, dim_embed)
            self.v_a = nn.Parameter(torch.randn(dim_embed))
            self.v_c = nn.Parameter(torch.randn(dim_embed))
        self.linear_1 = nn.Linear(dim_embed, dim_embed)
        self.linear_2 = nn.Linear(dim_embed, 1)
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
        return

    def forward(self, o):
        instance = o[1]
        projected_instance = self.project_s(instance)
        u_t = torch.matmul(self.v_a, torch.tanh(self.w_a(projected_instance
            )).permute(0, 2, 1))
        a_t = F.softmax(u_t, dim=1).unsqueeze(2).repeat(1, 1, self.dim_embed)
        c_t = a_t * projected_instance
        hidden_2 = torch.cat((projected_instance, c_t), dim=2)
        u_t_2 = torch.matmul(self.v_c, torch.tanh(self.w_c(hidden_2)).
            permute(0, 2, 1))
        prob = torch.softmax(u_t_2, dim=1)
        h_i = torch.bmm(prob.unsqueeze(1), projected_instance).squeeze(1)
        output_1 = F.relu(self.linear_1(h_i))
        v = self.linear_2(output_1).squeeze(1)
        return v


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'batch_size': 4, 'n_nodes': 4, 'dim_s': 4, 'dim_embed': 4}]
