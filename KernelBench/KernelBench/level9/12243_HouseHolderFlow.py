import torch
import torch.utils.data
import torch.nn as nn


class HouseHolderFlow(nn.Module):

    def forward(self, v, z):
        """
        :param v: batch_size (B) x latent_size (L)
        :param z: batch_size (B) x latent_size (L)
        :return: z_new = z - 2* v v_T / norm(v,2) * z
        """
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))
        vvTz = torch.bmm(vvT, z.unsqueeze(2)).squeeze(2)
        norm_sq = torch.sum(v * v, 1).unsqueeze(1)
        norm_sq = norm_sq.expand(norm_sq.size(0), v.size(1))
        z_new = z - 2 * vvTz / norm_sq
        return z_new


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
