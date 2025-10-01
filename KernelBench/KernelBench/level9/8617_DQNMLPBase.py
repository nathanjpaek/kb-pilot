from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


class ListModule(nn.Module):

    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class DQNMLPBase(nn.Module):

    def __init__(self, input_sz, num_actions, opt):
        super(DQNMLPBase, self).__init__()

        def init_(m):
            return init(m, init_normc_, lambda x: nn.init.constant_(x, 0))
        hid_sz = opt['hid_sz']
        num_layer = opt['num_layer']
        self.tanh = nn.Tanh()
        self.in_fc = init_(nn.Linear(input_sz, hid_sz))
        assert num_layer >= 1
        hid_layers = []
        for i in range(0, num_layer - 1):
            hid_fc = init_(nn.Linear(hid_sz, hid_sz))
            hid_layers.append(hid_fc)
        self.hid_layers = ListModule(*hid_layers)
        self.out_fc = init_(nn.Linear(hid_sz, num_actions))

    def forward(self, input):
        x = self.in_fc(input)
        x = self.tanh(x)
        for hid_fc in self.hid_layers:
            x = hid_fc(x)
            x = self.tanh(x)
        x = self.out_fc(x)
        return x

    @property
    def state_size(self):
        return 1


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_sz': 4, 'num_actions': 4, 'opt': _mock_config(
        hid_sz=4, num_layer=1)}]
