import torch
from torch import nn


def get_EF(input_size, dim, method='learnable', head_dim=None, bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Includes a method for convolution, as well as a method for no additional params.
    """
    assert method == 'learnable' or method == 'convolution' or method == 'no_params', "The method flag needs to be either 'learnable', 'convolution', or 'no_params'!"
    if method == 'convolution':
        conv = nn.Conv1d(head_dim, head_dim, kernel_size=int(input_size /
            dim), stride=int(input_size / dim))
        return conv
    if method == 'no_params':
        mat = torch.zeros((input_size, dim))
        torch.nn.init.normal_(mat, mean=0.0, std=1 / dim)
        return mat
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin


class linformerAttention(nn.Module):

    def __init__(self, dim, dropout, input_size, dim_k=20, full_attention=
        False, parameter_sharing=None):
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.dim_k = dim_k
        self.full_attention = full_attention
        self.input_size = input_size
        self.print_dim = False
        self.E = get_EF(input_size, dim=self.dim_k, method='learnable',
            head_dim=self.dim)
        self.F = (get_EF(input_size, dim=self.dim_k, method='learnable',
            head_dim=self.dim) if parameter_sharing == 'none' or 
            parameter_sharing == 'headwise' else self.E)
        self.is_proj_tensor = isinstance(self.E, torch.Tensor)

    def forward(self, q, k, v):
        if self.print_dim:
            None
            None
            None
        if not self.full_attention:
            if self.is_proj_tensor:
                self.E = self.E
                b, h, *_ = q.shape
                projection_E = repeat(self.E, 'j d -> b h j d', b=b, h=h)
                k = torch.einsum('...di,...dj->...ij', k, projection_E)
            else:
                k = torch.einsum('...ij->...ji', k)
                k = self.E(k)
        if self.print_dim:
            None
            None
            None
        q = torch.einsum('...id,...dj->...ij', q, k)
        P_bar = q / torch.sqrt(torch.tensor(self.dim_k).type(q.type()))
        P_bar = P_bar.softmax(dim=-1)
        P_bar = self.dropout(P_bar)
        if not self.full_attention:
            if self.is_proj_tensor:
                self.F = self.F
                v = torch.matmul(v, self.F)
            else:
                v = torch.einsum('...ij->...ji', v)
                v = self.F(v)
        out = torch.einsum('...id,...jd->...ij', P_bar, v)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'dropout': 0.5, 'input_size': 4}]
