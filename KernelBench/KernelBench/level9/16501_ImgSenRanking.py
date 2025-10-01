import torch
import numpy as np
import torch.utils.data


def l2norm(input, p=2.0, dim=1, eps=1e-12):
    """
    Compute L2 norm, row-wise
    """
    l2_inp = input / input.norm(p, dim, keepdim=True).clamp(min=eps)
    return l2_inp.expand_as(input)


def xavier_weight(tensor):
    nin, nout = tensor.size()[0], tensor.size()[1]
    r = np.sqrt(6.0) / np.sqrt(nin + nout)
    return tensor.normal_(0, r)


class ImgSenRanking(torch.nn.Module):

    def __init__(self, dim_image, sent_dim, hid_dim):
        super(ImgSenRanking, self).__init__()
        self.register_buffer('device_id', torch.IntTensor(1))
        self.linear_img = torch.nn.Linear(dim_image, hid_dim)
        self.linear_sent = torch.nn.Linear(sent_dim, hid_dim)
        self.init_weights()

    def init_weights(self):
        xavier_weight(self.linear_img.weight.data)
        xavier_weight(self.linear_sent.weight.data)
        self.linear_img.bias.data.fill_(0)
        self.linear_sent.bias.data.fill_(0)

    def forward(self, sent, img):
        img_vec = self.linear_img(img)
        sent_vec = self.linear_sent(sent)
        return l2norm(sent_vec), l2norm(img_vec)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_image': 4, 'sent_dim': 4, 'hid_dim': 4}]
