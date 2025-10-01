import torch
import numpy as np


def Interction_Union(outputs, targets):
    width_o = outputs[:, 2]
    width_t = targets[:, 2]
    height_o = outputs[:, 3]
    height_t = targets[:, 3]
    x_max = torch.max(torch.stack((outputs[:, 0] + outputs[:, 2] / 2, 
        targets[:, 0] + targets[:, 2] / 2), 1), 1)[0]
    x_min = torch.min(torch.stack((outputs[:, 0] - outputs[:, 2] / 2, 
        targets[:, 0] - targets[:, 2] / 2), 1), 1)[0]
    y_max = torch.max(torch.stack((outputs[:, 1] + outputs[:, 3] / 2, 
        targets[:, 1] + targets[:, 3] / 2), 1), 1)[0]
    y_min = torch.min(torch.stack((outputs[:, 1] - outputs[:, 3] / 2, 
        targets[:, 1] - targets[:, 3] / 2), 1), 1)[0]
    Area_o = torch.mul(width_o, height_o)
    Area_t = torch.mul(width_t, height_t)
    Inter_w = torch.add(width_o, width_t).sub(x_max.sub(x_min))
    Inter_t = torch.add(height_o, height_t).sub(y_max.sub(y_min))
    Inter = torch.mul(Inter_w, Inter_t)
    zeros = torch.zeros_like(Inter)
    Inter = torch.where(Inter < 0, zeros, Inter)
    Union = torch.add(Area_o, Area_t).sub(Inter)
    return Inter, Union, x_max, x_min, y_max, y_min


def Center_points(outputs, targets):
    x_o = outputs[:, 0]
    y_o = outputs[:, 1]
    x_t = targets[:, 0]
    y_t = targets[:, 1]
    return x_o, y_o, x_t, y_t


class DIoU_loss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        Inter, Union, x_max, x_min, y_max, y_min = Interction_Union(outputs,
            targets)
        IoU = torch.div(Inter, Union)
        C_width = x_max.sub(x_min)
        C_height = y_max.sub(y_min)
        C = torch.mul(C_width, C_height)
        x_o, y_o, x_t, y_t = Center_points(outputs, targets)
        dis = torch.add(torch.pow(x_o.sub(x_t), 2), torch.pow(y_o.sub(y_t), 2))
        R_DIoU = torch.div(dis, torch.pow(C, 2))
        ones = torch.ones_like(IoU)
        loss = torch.add(ones.sub(IoU), R_DIoU)
        zeros = torch.zeros_like(loss)
        loss = torch.where(loss < 0, zeros, loss)
        return torch.sum(loss)


class CIoU_loss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        Inter, Union, _x_max, _x_min, _y_max, _y_min = Interction_Union(outputs
            , targets)
        IoU = torch.div(Inter, Union)
        loss_DIoU = DIoU_loss()
        loss = loss_DIoU(outputs, targets)
        width_o = outputs[:, 2]
        width_t = targets[:, 2]
        height_o = outputs[:, 3]
        height_t = targets[:, 3]
        v = torch.pow(torch.arctan(torch.div(width_t, height_t)).sub(torch.
            arctan(torch.div(width_o, height_o))), 2) * 4 / (np.pi * np.pi)
        alpha = torch.div(v, 1 + v.sub(IoU))
        R_CIoU = torch.mul(alpha, v)
        loss = torch.add(loss, R_CIoU)
        zeros = torch.zeros_like(loss)
        loss = torch.where(loss < 0, zeros, loss)
        return torch.sum(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
