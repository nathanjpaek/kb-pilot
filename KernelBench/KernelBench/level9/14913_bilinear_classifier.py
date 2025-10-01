import torch
import torch.nn as nn
import torch.utils.data.dataloader
import torch.nn


class Sparse_dropout(nn.Module):

    def __init__(self, p):
        super(Sparse_dropout, self).__init__()
        self.dropout_rate = p

    def forward(self, input, noise_shape):
        if not self.training:
            return input
        shapes = input.shape
        noise_shape = list(noise_shape)
        broadcast_dims = []
        for idx, dim_pair in enumerate(zip(shapes, noise_shape)):
            if dim_pair[1] > 1:
                broadcast_dims.append((idx, dim_pair[0]))
        mask_dims = []
        for dim in broadcast_dims:
            mask_dims.append(dim[1])
        mask = torch.bernoulli((torch.ones(mask_dims, device=input.device) *
            (1 - self.dropout_rate)).reshape(noise_shape)) * (1 / (1 - self
            .dropout_rate))
        mask
        return input * mask


class biaffine_mapping(nn.Module):

    def __init__(self, input_size_x, input_size_y, output_size, bias_x,
        bias_y, initializer=None):
        super(biaffine_mapping, self).__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.output_size = output_size
        self.initilizer = None
        if self.bias_x:
            input_size1 = input_size_x + 1
            input_size2 = input_size_y + 1
        self.biaffine_map = nn.Parameter(torch.Tensor(input_size1,
            output_size, input_size2))
        self.initialize()

    def initialize(self):
        if self.initilizer is None:
            torch.nn.init.orthogonal_(self.biaffine_map)
        else:
            self.initilizer(self.biaffine_map)

    def forward(self, x, y):
        batch_size, bucket_size = x.shape[0], x.shape[1]
        if self.bias_x:
            x = torch.cat([x, torch.ones([batch_size, bucket_size, 1],
                device=x.device)], axis=2)
        if self.bias_y:
            y = torch.cat([y, torch.ones([batch_size, bucket_size, 1],
                device=y.device)], axis=2)
        x_set_size, y_set_size = x.shape[-1], y.shape[-1]
        x = x.reshape(-1, x_set_size)
        biaffine_map = self.biaffine_map.reshape(x_set_size, -1)
        biaffine_mapping = torch.matmul(x, biaffine_map).reshape(batch_size,
            -1, y_set_size)
        biaffine_mapping = biaffine_mapping.bmm(torch.transpose(y, 1, 2)
            ).reshape(batch_size, bucket_size, self.output_size, bucket_size)
        biaffine_mapping = biaffine_mapping.transpose(2, 3)
        return biaffine_mapping


class bilinear_classifier(nn.Module):

    def __init__(self, dropout, input_size_x, input_size_y, output_size,
        bias_x=True, bias_y=True):
        super(bilinear_classifier, self).__init__()
        self.dropout_rate = dropout
        self.output_size = output_size
        self.dropout = Sparse_dropout(p=self.dropout_rate)
        self.biaffine = biaffine_mapping(input_size_x, input_size_y,
            output_size, bias_x, bias_y)

    def forward(self, x_bnv, y_bnv):
        batch_size, input_size_x = x_bnv.shape[0], x_bnv.shape[-1]
        input_size_y = y_bnv.shape[-1]
        noise_shape_x = [batch_size, 1, input_size_x]
        noise_shape_y = [batch_size, 1, input_size_y]
        x = self.dropout(x_bnv, noise_shape_x)
        y = self.dropout(y_bnv, noise_shape_y)
        output = self.biaffine(x, y)
        if self.output_size == 1:
            output = output.squeeze(-1)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dropout': 0.5, 'input_size_x': 4, 'input_size_y': 4,
        'output_size': 4}]
