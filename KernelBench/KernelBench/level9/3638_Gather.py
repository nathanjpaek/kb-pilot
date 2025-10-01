import torch
import torch.nn as nn


class Gather(torch.nn.Module):
    """ 
        gather
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, F, K, use_mask=False):
        super().__init__()
        self.K = K
        self.F = F
        self.softmax_2 = nn.Softmax(dim=2)
        self.use_mask = use_mask
        if use_mask:
            self.a = nn.Parameter(torch.randn(F, K), requires_grad=True)
    """image_codeing [N,F,H,W]
        s_att (spatial attention): [N,K,H,W]     K-how many segment     s_att(0 or 50)
        feature: [N,K,F]
    """

    def forward(self, image_coding, s_att, att_mask=None):
        """
            x: b*m*h*w
            c_att: b*K*h*w
        """
        b, F, h, w = image_coding.size()
        b_, K, h_, w_ = s_att.size()
        assert b == b_ and h == h_ and w == w_ and self.K == K and self.F == F
        if self.use_mask:
            b__, K__ = att_mask.size()
            assert b == b__ and self.K == K__
        s_att_new = self.softmax_2(s_att.view(b, K, h * w)).permute(0, 2, 1)
        gather_result = torch.bmm(image_coding.view(b, F, h * w), s_att_new)
        if self.use_mask:
            att_mask_new = att_mask.view(b__, 1, K__).expand_as(gather_result)
            gather_result = att_mask_new * gather_result + (1 - att_mask_new
                ) * self.a.view(1, F, K).expand_as(gather_result)
        return gather_result


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'F': 4, 'K': 4}]
