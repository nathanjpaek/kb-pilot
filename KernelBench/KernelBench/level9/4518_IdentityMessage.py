import torch
import torch.utils.data


class IdentityMessage(torch.nn.Module):

    def __init__(self, raw_msg_dim: 'int', memory_dim: 'int', time_dim: 'int'):
        super(IdentityMessage, self).__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'raw_msg_dim': 4, 'memory_dim': 4, 'time_dim': 4}]
