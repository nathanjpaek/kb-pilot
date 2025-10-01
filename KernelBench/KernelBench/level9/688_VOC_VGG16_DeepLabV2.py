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


class ConvReLU(nn.Sequential):

    def __init__(self, in_ch, out_ch, dilation, layer_idx, seq_idx):
        super(ConvReLU, self).__init__()
        self.add_module(f'conv{layer_idx}_{seq_idx}', nn.Conv2d(in_channels
            =in_ch, out_channels=out_ch, kernel_size=3, padding=dilation,
            dilation=dilation))
        self.add_module(f'relu{layer_idx}_{seq_idx}', nn.ReLU(inplace=True))


class VGGLayer(nn.Sequential):

    def __init__(self, in_ch, out_ch, conv_num, dilation, pool_size,
        pool_stride, layer_idx):
        super(VGGLayer, self).__init__()
        for seq_idx in range(1, conv_num + 1):
            self.add_module(f'conv_relu_{seq_idx}', ConvReLU(in_ch=in_ch if
                seq_idx == 1 else out_ch, out_ch=out_ch, dilation=dilation,
                layer_idx=layer_idx, seq_idx=seq_idx))
        self.add_module(f'pool{layer_idx}', nn.MaxPool2d(kernel_size=
            pool_size, stride=pool_stride, padding=pool_size % 2, ceil_mode
            =True))


class VGGFeature(nn.Sequential):

    def __init__(self, in_ch, out_chs, conv_nums, dilations, pool_strides,
        pool_size):
        super(VGGFeature, self).__init__()
        for i, layer_idx in enumerate(range(1, len(out_chs) + 1)):
            self.add_module(f'layer{layer_idx}', VGGLayer(in_ch=in_ch if 
                layer_idx == 1 else out_chs[i - 1], out_ch=out_chs[i],
                conv_num=conv_nums[i], dilation=dilations[i], pool_size=
                pool_size, pool_stride=pool_strides[i], layer_idx=layer_idx))


class VOC_VGG16_DeepLabV2(nn.Module):

    def __init__(self):
        super(VOC_VGG16_DeepLabV2, self).__init__()
        self.VGGFeature = VGGFeature(in_ch=3, out_chs=[64, 128, 256, 512, 
            512], conv_nums=[2, 2, 3, 3, 3], dilations=[1, 1, 1, 1, 2],
            pool_strides=[2, 2, 2, 1, 1], pool_size=3)
        self.VGGASPP = VGGASPP(in_ch=512, num_classes=21, rates=[6, 12, 18,
            24], start_layer_idx=6, net_id='pascal')

    def forward(self, x):
        x = self.VGGFeature(x)
        return self.VGGASPP(x)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
