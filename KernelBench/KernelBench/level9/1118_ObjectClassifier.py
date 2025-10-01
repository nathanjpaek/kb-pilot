import torch
from torch import nn


class ObjectClassifier(nn.Module):
    """
    perform log likelihood over sequence data ie. log(softmax), permute dimension
      accordingly to meet NLLLoss requirement
    Input: [seq_len, bsz, d_input]
    Output: [bsz, num_classes, seq_len]

    Usage:
    bsz=5; seq=16; d_input=1024; num_classes=10
    classiifer = ObjectClassifier(d_input, num_classes)
    x = torch.rand(seq, bsz, d_input)  # 16x5x1024
    out = classifier(x)  # 5x10x16
    """

    def __init__(self, d_input, num_classes):
        super(ObjectClassifier, self).__init__()
        self.d_input = d_input
        self.num_classes = num_classes
        self.linear = nn.Linear(d_input, num_classes)
        self.classifier = nn.LogSoftmax(dim=1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        out = self.linear(x)
        out = out.permute(1, 2, 0)
        return self.classifier(out)

    def extra_repr(self) ->str:
        return 'SxBx%d -> Bx%dxS' % (self.d_input, self.num_classes)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_input': 4, 'num_classes': 4}]
