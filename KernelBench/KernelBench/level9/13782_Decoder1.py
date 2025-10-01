import torch
import torch.nn as nn


class Decoder1(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Decoder1, self).__init__()
        self.fixed = fixed
        self.conv11 = nn.Conv2d(64, 3, 3, 1, 0, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = load_lua(model)
                load_param(t7_model, 1, self.conv11)
            else:
                self.load_state_dict(torch.load(model, map_location=lambda
                    storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.relu(self.conv11(self.pad(input)))
        return y

    def forward_branch(self, input):
        out11 = self.relu(self.conv11(self.pad(input)))
        return out11,


def get_inputs():
    return [torch.rand([4, 64, 4, 4])]


def get_init_inputs():
    return [[], {}]
