import torch
from torch import nn


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the fc layer of the net.
    """
    b = [model.linear]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for the conv layer of the net.
    """
    b = [model.res2plus1d]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained=False):
        self._prepare_base_model()
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 
            1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),
            padding=(0, 1, 1))
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.__init_weight()
        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        logits = self.fc8(x)
        return logits

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {'features.0.weight': 'conv1.weight',
            'features.0.bias': 'conv1.bias', 'features.3.weight':
            'conv2.weight', 'features.3.bias': 'conv2.bias',
            'features.6.weight': 'conv3a.weight', 'features.6.bias':
            'conv3a.bias', 'features.8.weight': 'conv3b.weight',
            'features.8.bias': 'conv3b.bias', 'features.11.weight':
            'conv4a.weight', 'features.11.bias': 'conv4a.bias',
            'features.13.weight': 'conv4b.weight', 'features.13.bias':
            'conv4b.bias', 'features.16.weight': 'conv5a.weight',
            'features.16.bias': 'conv5a.bias', 'features.18.weight':
            'conv5b.weight', 'features.18.bias': 'conv5b.bias',
            'classifier.0.weight': 'fc6.weight', 'classifier.0.bias':
            'fc6.bias', 'classifier.3.weight': 'fc7.weight',
            'classifier.3.bias': 'fc7.bias'}
        p_dict = torch.load(pretrained_path)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_optim_policies(self, lr):
        return [{'params': get_1x_lr_params(self), 'lr': lr}, {'params':
            get_10x_lr_params(self), 'lr': lr * 10}]

    def _prepare_base_model(self):
        self.crop_size = 112
        self.scale_size = 256
        self.input_mean = [0.43216, 0.394666, 0.37645]
        self.input_std = [0.22803, 0.22145, 0.216989]


def get_inputs():
    return [torch.rand([4, 3, 64, 64, 64])]


def get_init_inputs():
    return [[], {'num_classes': 4}]
