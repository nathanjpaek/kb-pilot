import torch
from torch import nn
import torch.nn
import torch.optim


class PositionalEncoding(nn.Module):
    """
    A special, non-learnable positional encoding for handling variable (possibly longer)
    lengths of inputs. We simply add an ordinal number as an additional dimension for
    the input embeddings, and then project them back to the original number of dimensions
    """

    def __init__(self, dim_model):
        super().__init__()
        self.pos_embed = nn.Linear(dim_model + 1, dim_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        device = x.device
        batch_size, seq_len, _ = x.shape
        position_idx = torch.arange(0, seq_len, device=device).unsqueeze(0
            ).repeat(batch_size, 1).reshape(batch_size, seq_len, 1)
        x_pos = torch.cat((x, position_idx), dim=2)
        return self.activation(self.pos_embed(x_pos))


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_model': 4}]
