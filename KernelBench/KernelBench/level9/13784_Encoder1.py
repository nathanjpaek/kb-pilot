import torch
import torch.nn as nn


class Encoder1(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Encoder1, self).__init__()
        self.fixed = fixed
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 0, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = load_lua(model)
                load_param(t7_model, 0, self.conv0)
                load_param(t7_model, 2, self.conv11)
            else:
                self.load_state_dict(torch.load(model, map_location=lambda
                    storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.conv0(input)
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_branch(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        return out11,


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
