import torch
import torch.nn as nn


class Decoder5(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Decoder5, self).__init__()
        self.fixed = fixed
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv44 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv41 = nn.Conv2d(512, 256, 3, 1, 0)
        self.conv34 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv31 = nn.Conv2d(256, 128, 3, 1, 0)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv21 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv11 = nn.Conv2d(64, 3, 3, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = load_lua(model)
                load_param(t7_model, 1, self.conv51)
                load_param(t7_model, 5, self.conv44)
                load_param(t7_model, 8, self.conv43)
                load_param(t7_model, 11, self.conv42)
                load_param(t7_model, 14, self.conv41)
                load_param(t7_model, 18, self.conv34)
                load_param(t7_model, 21, self.conv33)
                load_param(t7_model, 24, self.conv32)
                load_param(t7_model, 27, self.conv31)
                load_param(t7_model, 31, self.conv22)
                load_param(t7_model, 34, self.conv21)
                load_param(t7_model, 38, self.conv12)
                load_param(t7_model, 41, self.conv11)
            else:
                self.load_state_dict(torch.load(model, map_location=lambda
                    storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.relu(self.conv51(self.pad(input)))
        y = self.unpool(y)
        y = self.relu(self.conv44(self.pad(y)))
        y = self.relu(self.conv43(self.pad(y)))
        y = self.relu(self.conv42(self.pad(y)))
        y = self.relu(self.conv41(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv34(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv31(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_branch(self, input):
        out51 = self.relu(self.conv51(self.pad(input)))
        out51 = self.unpool(out51)
        out44 = self.relu(self.conv44(self.pad(out51)))
        out43 = self.relu(self.conv43(self.pad(out44)))
        out42 = self.relu(self.conv42(self.pad(out43)))
        out41 = self.relu(self.conv41(self.pad(out42)))
        out41 = self.unpool(out41)
        out34 = self.relu(self.conv34(self.pad(out41)))
        out33 = self.relu(self.conv33(self.pad(out34)))
        out32 = self.relu(self.conv32(self.pad(out33)))
        out31 = self.relu(self.conv31(self.pad(out32)))
        out31 = self.unpool(out31)
        out22 = self.relu(self.conv22(self.pad(out31)))
        out21 = self.relu(self.conv21(self.pad(out22)))
        out21 = self.unpool(out21)
        out12 = self.relu(self.conv12(self.pad(out21)))
        out11 = self.relu(self.conv11(self.pad(out12)))
        return out51, out41, out31, out21, out11


def get_inputs():
    return [torch.rand([4, 512, 4, 4])]


def get_init_inputs():
    return [[], {}]
