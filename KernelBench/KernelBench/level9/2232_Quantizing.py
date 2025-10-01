import torch
import torch.nn as nn
from typing import Tuple


class Quantizing(nn.Module):
    """
    This is quantizing layer.
    """
    __initialized: 'bool' = True

    def __init__(self, num_quantizing: 'int', quantizing_dim: 'int',
        _weight: 'torch.Tensor'=None, initialize_by_dataset: 'bool'=True,
        mean: 'float'=0.0, std: 'float'=1.0, dtype: 'torch.dtype'=None,
        device: 'torch.device'=None):
        super().__init__()
        assert num_quantizing > 0
        assert quantizing_dim > 0
        self.num_quantizing = num_quantizing
        self.quantizing_dim = quantizing_dim
        self.initialize_by_dataset = initialize_by_dataset
        self.mean, self.std = mean, std
        if _weight is None:
            self.weight = nn.Parameter(torch.empty(num_quantizing,
                quantizing_dim, dtype=dtype, device=device))
            nn.init.normal_(self.weight, mean=mean, std=std)
            if initialize_by_dataset:
                self.__initialized = False
                self.__initialized_length = 0
        else:
            assert _weight.dim() == 2
            assert _weight.size(0) == num_quantizing
            assert _weight.size(1) == quantizing_dim
            self.weight = nn.Parameter(_weight.to(device))

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor]:
        """
        x   : shape is (*, E), and weight shape is (Q, E). 
        return -> ( quantized : shape is (*, E), quantized_idx : shape is (*,) )
        """
        input_size = x.shape
        h = x.view(-1, self.quantizing_dim)
        if not self.__initialized and self.initialize_by_dataset:
            getting_len = self.num_quantizing - self.__initialized_length
            init_weight = h[torch.randperm(len(h))[:getting_len]]
            _until = self.__initialized_length + init_weight.size(0)
            self.weight.data[self.__initialized_length:_until] = init_weight
            self.__initialized_length = _until
            None
            if _until >= self.num_quantizing:
                self.__initialized = True
                None
        delta = self.weight.unsqueeze(0) - h.unsqueeze(1)
        dist = torch.sum(delta * delta, dim=-1)
        q_idx = torch.argmin(dist, dim=-1)
        q_data = self.weight[q_idx]
        return q_data.view(input_size), q_idx.view(input_size[:1])

    def from_idx(self, idx: 'torch.Tensor') ->torch.Tensor:
        """
        idx: shape is (*, ). int tensor.
        return -> (*, E) float tensor
        """
        input_size = idx.shape
        i = idx.view(-1)
        q_data = self.weight[i].view(*input_size, self.quantizing_dim)
        return q_data

    def load_state_dict(self, state_dict, strict: 'bool'):
        self.__initialized = True
        return super().load_state_dict(state_dict, strict=strict)

    def __repr__(self):
        s = f'Quantizing({self.num_quantizing}, {self.quantizing_dim})'
        return s

    def isInitialized(self) ->bool:
        return self.__initialized


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_quantizing': 4, 'quantizing_dim': 4}]
