import torch


class node_norm(torch.nn.Module):

    def __init__(self, node_norm_type='n', unbiased=False, eps=1e-05,
        power_root=2, **kwargs):
        super(node_norm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps
        self.node_norm_type = node_norm_type
        self.power = 1 / power_root

    def forward(self, x):
        if self.node_norm_type == 'n':
            mean = torch.mean(x, dim=1, keepdim=True)
            std = (torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True
                ) + self.eps).sqrt()
            x = (x - mean) / std
        elif self.node_norm_type == 'v':
            std = (torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True
                ) + self.eps).sqrt()
            x = x / std
        elif self.node_norm_type == 'm':
            mean = torch.mean(x, dim=1, keepdim=True)
            x = x - mean
        elif self.node_norm_type == 'srv':
            std = (torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True
                ) + self.eps).sqrt()
            x = x / torch.sqrt(std)
        elif self.node_norm_type == 'pr':
            std = (torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True
                ) + self.eps).sqrt()
            x = x / torch.pow(std, self.power)
        return x

    def __repr__(self):
        original_str = super().__repr__()
        components = list(original_str)
        node_norm_type_str = f'node_norm_type={self.node_norm_type}'
        components.insert(-1, node_norm_type_str)
        new_str = ''.join(components)
        return new_str


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
