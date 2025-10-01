import torch


class Reconstruction_Layer(torch.nn.Module):
    """TThis is the reconstruction layer for the network to learn how to remake
    the original input image"""

    def __init__(self, batch_size, capsin_n_maps, capsin_n_dims,
        reconstruct_channels):
        super(Reconstruction_Layer, self).__init__()
        self.batch_size = batch_size
        self.capsin_n_dims = capsin_n_dims
        self.capsin_n_maps = capsin_n_maps
        self.reconstruct_channels = reconstruct_channels
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv1_params = {'i': int(self.capsin_n_maps * self.
            capsin_n_dims), 'o': 64, 'k': 1, 's': 1, 'p': 0}
        self.conv1 = torch.nn.Conv2d(in_channels=self.conv1_params['i'],
            out_channels=self.conv1_params['o'], kernel_size=self.
            conv1_params['k'], stride=self.conv1_params['s'], padding=self.
            conv1_params['p'])
        self.conv2_params = {'i': int(self.conv1_params['o']), 'o': 128,
            'k': 1, 's': 1, 'p': 0}
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv2_params['i'],
            out_channels=self.conv2_params['o'], kernel_size=self.
            conv2_params['k'], stride=self.conv2_params['s'], padding=self.
            conv2_params['p'])
        self.conv3_params = {'i': int(self.conv2_params['o']), 'o': self.
            reconstruct_channels, 'k': 1, 's': 1, 'p': 0}
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv3_params['i'],
            out_channels=self.conv3_params['o'], kernel_size=self.
            conv3_params['k'], stride=self.conv3_params['s'], padding=self.
            conv3_params['p'])

    def forward(self, x):
        _, _, h, w, _ = x.size()
        x = x.permute(0, 1, 4, 2, 3)
        x = x.contiguous().view([-1, self.capsin_n_maps * self.
            capsin_n_dims, h, w])
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'batch_size': 4, 'capsin_n_maps': 4, 'capsin_n_dims': 4,
        'reconstruct_channels': 4}]
