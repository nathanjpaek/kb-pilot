import torch
import torch.nn as nn


class Encoder2(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Encoder2, self).__init__()
        self.fixed = fixed
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = load_lua(model)
                load_param(t7_model, 0, self.conv0)
                load_param(t7_model, 2, self.conv11)
                load_param(t7_model, 5, self.conv12)
                load_param(t7_model, 9, self.conv21)
            else:
                self.load_state_dict(torch.load(model, map_location=lambda
                    storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.conv0(input)
        y = self.relu(self.conv11(self.pad(y)))
        y = self.relu(self.conv12(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv21(self.pad(y)))
        return y

    def forward_branch(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        return out11, out21


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
