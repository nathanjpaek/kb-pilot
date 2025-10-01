import torch
import torch.nn as nn
import torch.nn.functional as F


class RPN_Up(nn.Module):
    """
    For SiamRPN
    """

    def __init__(self, anchor_nums=5, inchannels=256, outchannels=256,
        cls_type='thicker'):
        super(RPN_Up, self).__init__()
        self.anchor_nums = anchor_nums
        self.inchannels = inchannels
        self.outchannels = outchannels
        if cls_type == 'thinner':
            self.cls_channel = self.anchor_nums
        elif cls_type == 'thicker':
            self.cls_channel = self.anchor_nums * 2
        else:
            raise ValueError('not implemented cls/loss type')
        self.reg_channel = 4 * self.anchor_nums
        self.template_cls = nn.Conv2d(self.inchannels, self.outchannels *
            self.cls_channel, kernel_size=3)
        self.template_reg = nn.Conv2d(self.inchannels, self.outchannels *
            self.reg_channel, kernel_size=3)
        self.search_cls = nn.Conv2d(self.inchannels, self.outchannels,
            kernel_size=3)
        self.search_reg = nn.Conv2d(self.inchannels, self.outchannels,
            kernel_size=3)
        self.adjust = nn.Conv2d(self.reg_channel, self.reg_channel,
            kernel_size=1)

    def _conv2d_group(self, x, kernel):
        batch = kernel.size()[0]
        pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
        px = x.view(1, -1, x.size()[2], x.size()[3])
        po = F.conv2d(px, pk, groups=batch)
        po = po.view(batch, -1, po.size()[2], po.size()[3])
        return po

    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls(z_f)
        reg_kernel = self.template_reg(z_f)
        cls_feature = self.search_cls(x_f)
        loc_feature = self.search_reg(x_f)
        _, _, _s_cls, _ = cls_kernel.size()
        _, _, _s_reg, _ = reg_kernel.size()
        pred_cls = self._conv2d_group(cls_feature, cls_kernel)
        pred_reg = self.adjust(self._conv2d_group(loc_feature, reg_kernel))
        return pred_cls, pred_reg


def get_inputs():
    return [torch.rand([4, 256, 64, 64]), torch.rand([4, 256, 64, 64])]


def get_init_inputs():
    return [[], {}]
