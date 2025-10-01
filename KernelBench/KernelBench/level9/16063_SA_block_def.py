import torch
import torch.nn as nn


class SA_block_def(nn.Module):
    """Self-Attention block with dot product for point/voxel/pillar context.
    """

    def __init__(self, inplanes, planes, groups=4):
        super().__init__()
        self.groups = groups
        self.t = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.p = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.g = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.z = nn.Conv1d(planes, inplanes, kernel_size=1, stride=1,
            groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)
        self.softmax = nn.Softmax(dim=-1)

    def kernel(self, t, p, g, b, c, h):
        """Return the output after dot product per head
        Args:
            t: output of linear value
            p: output of linear query
            g: output of linear keys
            b: batch size
            c: no of channels
            h: spatial breadth of feature maps
        """
        proj_query = p.permute(0, 2, 1)
        proj_key = g
        energy = torch.bmm(proj_query, proj_key)
        total_energy = energy
        attention = self.softmax(total_energy)
        proj_value = t
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        return out

    def forward(self, x, y):
        residual = x
        t = self.t(y)
        p = self.p(x)
        g = self.g(y)
        b, c, h = t.size()
        if self.groups and self.groups > 1:
            _c = int(c / self.groups)
            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)
            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h)
                _t_sequences.append(_x)
            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, h)
        x = self.z(x)
        x = self.gn(x) + residual
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
