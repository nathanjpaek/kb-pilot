import torch


class rpn_head(torch.nn.Module):

    def __init__(self, in_channels=1024, out_channels=1024, n_anchors=15):
        super(rpn_head, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_rpn = torch.nn.Conv2d(in_channels, out_channels, 3,
            stride=1, padding=1)
        self.rpn_cls_prob = torch.nn.Conv2d(out_channels, n_anchors, 1,
            stride=1, padding=0)
        self.rpn_bbox_pred = torch.nn.Conv2d(out_channels, 4 * n_anchors, 1,
            stride=1, padding=0)

    def forward(self, x):
        conv_rpn = self.relu(self.conv_rpn(x))
        rpn_cls_prob = self.sigmoid(self.rpn_cls_prob(conv_rpn))
        rpn_bbox_pred = self.rpn_bbox_pred(conv_rpn)
        return rpn_cls_prob, rpn_bbox_pred


def get_inputs():
    return [torch.rand([4, 1024, 64, 64])]


def get_init_inputs():
    return [[], {}]
