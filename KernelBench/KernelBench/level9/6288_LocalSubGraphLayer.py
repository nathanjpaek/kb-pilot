import torch
from torch import nn
import torch.nn.functional as F


class LocalMLP(nn.Module):

    def __init__(self, dim_in: 'int', use_norm: 'bool'=True):
        """a Local 1 layer MLP

        :param dim_in: feat in size
        :type dim_in: int
        :param use_norm: if to apply layer norm, defaults to True
        :type use_norm: bool, optional
        """
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_in, bias=not use_norm)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(dim_in)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """forward of the module

        :param x: input tensor (..., dim_in)
        :type x: torch.Tensor
        :return: output tensor (..., dim_in)
        :rtype: torch.Tensor
        """
        x = self.linear(x)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        x = F.relu(x, inplace=True)
        return x


class LocalSubGraphLayer(nn.Module):

    def __init__(self, dim_in: 'int', dim_out: 'int') ->None:
        """Local subgraph layer

        :param dim_in: input feat size
        :type dim_in: int
        :param dim_out: output feat size
        :type dim_out: int
        """
        super(LocalSubGraphLayer, self).__init__()
        self.mlp = LocalMLP(dim_in)
        self.linear_remap = nn.Linear(dim_in * 2, dim_out)

    def forward(self, x: 'torch.Tensor', invalid_mask: 'torch.Tensor'
        ) ->torch.Tensor:
        """Forward of the model

        :param x: input tensor
        :tensor (B,N,P,dim_in)
        :param invalid_mask: invalid mask for x
        :tensor invalid_mask (B,N,P)
        :return: output tensor (B,N,P,dim_out)
        :rtype: torch.Tensor
        """
        _, num_vectors, _ = x.shape
        x = self.mlp(x)
        masked_x = x.masked_fill(invalid_mask[..., None] > 0, float('-inf'))
        x_agg = masked_x.max(dim=1, keepdim=True).values
        x_agg = x_agg.repeat(1, num_vectors, 1)
        x = torch.cat([x, x_agg], dim=-1)
        x = self.linear_remap(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4}]
