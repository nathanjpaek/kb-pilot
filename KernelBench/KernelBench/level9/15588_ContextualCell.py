from _paritybench_helpers import _mock_config
import torch
from torch import nn


def conv_bn_relu(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.Sequential(nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
        padding=padding, bias=False), nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(inplace=False))


class AggregateCell(nn.Module):
    """
        before aggregating, both paths should undergo
        conv1x1 with the output channels being equal to the smallest channel size among them
        upsampled to the highest resolution among them
        pre_transform: whether to do convbnrelu before summing up
    """

    def __init__(self, size_1, size_2, agg_size, pre_transform=True):
        super(AggregateCell, self).__init__()
        self.pre_transform = pre_transform
        if self.pre_transform:
            self.branch_1 = conv_bn_relu(size_1, agg_size, 1, 1, 0)
            self.branch_2 = conv_bn_relu(size_2, agg_size, 1, 1, 0)

    def forward(self, x1, x2):
        if self.pre_transform:
            x1 = self.branch_1(x1)
            x2 = self.branch_2(x2)
        if x1.size()[2:] > x2.size()[2:]:
            x2 = nn.Upsample(size=x1.size()[2:], mode='bilinear')(x2)
        elif x1.size()[2:] < x2.size()[2:]:
            x1 = nn.Upsample(size=x2.size()[2:], mode='bilinear')(x1)
        return x1 + x2


class ContextualCell(nn.Module):
    """New contextual cell design

    Config contains [op1, [loc1, loc2, op1, op2], [...], [...]]

    """

    def __init__(self, config, inp, repeats=1):
        super(ContextualCell, self).__init__()
        self._ops = nn.ModuleList()
        self._pos = []
        self._collect_inds = [0]
        self._pools = ['x']
        for ind, op in enumerate(config):
            if ind == 0:
                pos = 0
                op_id = op
                self._collect_inds.remove(pos)
                op_name = OP_NAMES[op_id]
                self._ops.append(OPS[op_name](inp, inp, 1, True, repeats))
                self._pos.append(pos)
                self._collect_inds.append(ind + 1)
                self._pools.append('{}({})'.format(op_name, self._pools[pos]))
            else:
                pos1, pos2, op_id1, op_id2 = op
                for pos, op_id in zip([pos1, pos2], [op_id1, op_id2]):
                    if pos in self._collect_inds:
                        self._collect_inds.remove(pos)
                    op_name = OP_NAMES[op_id]
                    self._ops.append(OPS[op_name](inp, inp, 1, True, repeats))
                    self._pos.append(pos)
                    self._pools.append('{}({})'.format(op_name, self._pools
                        [pos]))
                op_name = 'sum'
                self._ops.append(AggregateCell(size_1=None, size_2=None,
                    agg_size=inp, pre_transform=False))
                self._pos.append([ind * 3 - 1, ind * 3])
                self._collect_inds.append(ind * 3 + 1)
                self._pools.append('{}({},{})'.format(op_name, self._pools[
                    ind * 3 - 1], self._pools[ind * 3]))

    def forward(self, x):
        feats = [x]
        for pos, op in zip(self._pos, self._ops):
            if isinstance(pos, list):
                assert len(pos) == 2, 'Two ops must be provided'
                feats.append(op(feats[pos[0]], feats[pos[1]]))
            else:
                feats.append(op(feats[pos]))
        out = 0
        for i in self._collect_inds:
            out += feats[i]
        return out

    def prettify(self):
        return ' + '.join(self._pools[i] for i in self._collect_inds)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(), 'inp': 4}]
