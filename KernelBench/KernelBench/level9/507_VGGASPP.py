import torch
from torch import nn


class FCReLUDrop(nn.Sequential):

    def __init__(self, in_ch, out_ch, kernel_size, dilation, padding,
        layer_idx, branch_idx):
        super(FCReLUDrop, self).__init__()
        self.add_module(f'fc{layer_idx}_{branch_idx}', nn.Conv2d(in_ch,
            out_ch, kernel_size, stride=1, padding=padding, dilation=dilation))
        self.add_module(f'relu{layer_idx}_{branch_idx}', nn.ReLU(inplace=True))
        self.add_module(f'drop{layer_idx}_{branch_idx}', nn.Dropout(p=0.5))


class VGGASPPBranch(nn.Sequential):

    def __init__(self, in_ch, num_classes, rate, start_layer_idx,
        branch_idx, net_id):
        super(VGGASPPBranch, self).__init__()
        self.add_module(f'aspp_layer{start_layer_idx}_{branch_idx}',
            FCReLUDrop(in_ch, out_ch=1024, kernel_size=3, dilation=rate,
            padding=rate, layer_idx=start_layer_idx, branch_idx=branch_idx))
        self.add_module(f'aspp_layer{start_layer_idx + 1}_{branch_idx}',
            FCReLUDrop(in_ch=1024, out_ch=1024, kernel_size=1, dilation=1,
            padding=0, layer_idx=start_layer_idx + 1, branch_idx=branch_idx))
        self.add_module(f'fc{start_layer_idx + 2}_{net_id}_{branch_idx}',
            nn.Conv2d(in_channels=1024, out_channels=num_classes,
            kernel_size=1))
        fc_logit = eval('self.' +
            f'fc{start_layer_idx + 2}_{net_id}_{branch_idx}')
        nn.init.normal_(fc_logit.weight, mean=0.0, std=0.01)
        nn.init.constant_(fc_logit.bias, 0.0)


class VGGASPP(nn.Module):

    def __init__(self, in_ch, num_classes, rates, start_layer_idx, net_id=
        'pascal'):
        super(VGGASPP, self).__init__()
        for rate, branch_idx in zip(rates, range(1, len(rates) + 1)):
            self.add_module(f'aspp_branch{branch_idx}', VGGASPPBranch(in_ch,
                num_classes, rate, start_layer_idx, branch_idx, net_id))

    def forward(self, x):
        return sum([branch(x) for branch in self.children()])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'num_classes': 4, 'rates': [4, 4],
        'start_layer_idx': 1}]
