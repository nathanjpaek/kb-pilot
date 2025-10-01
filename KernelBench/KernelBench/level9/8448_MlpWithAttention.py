import torch
import torch.nn as nn


class Self_Attn1D(nn.Module):
    """ Self attention Layer """

    def __init__(self, in_dim, activation, k=8):
        super(Self_Attn1D, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim //
            k, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim //
            k, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim,
            kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_attn=False):
        """
            inputs :
                x : input feature maps(B X C X T)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*T)
        """
        B, C = x.size()
        T = 1
        x = x.view(B, C, T)
        proj_query = self.query_conv(x).view(B, -1, T).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, T)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, T)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, T)
        out = self.gamma * out + x
        out = out.squeeze(2)
        return out, attention


class MlpWithAttention(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MlpWithAttention, self).__init__()
        out = max(8, in_dim * 2)
        self.input = nn.Linear(in_dim, out)
        self.output = nn.Linear(out, out_dim)
        self.fc = nn.Linear(out, out)
        self.fc2 = nn.Linear(out, out)
        self.fc3 = nn.Linear(out, out)
        self.attention = Self_Attn1D(out, nn.LeakyReLU)
        self.attention2 = Self_Attn1D(out, nn.LeakyReLU)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.float()
        x = self.relu(self.input(x))
        x, _ = self.attention(x)
        x = self.relu(self.fc(x))
        x, _ = self.attention2(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.output(x))
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
