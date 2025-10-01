import torch


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


class GIoU_loss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        Inter, Union, x_max, x_min, y_max, y_min = Interction_Union(outputs,
            targets)
        IoU = torch.div(Inter, Union)
        C_width = x_max.sub(x_min)
        C_height = y_max.sub(y_min)
        C = torch.mul(C_width, C_height)
        GIoU = IoU.sub(torch.div(C.sub(Union), C))
        ones = torch.ones_like(GIoU)
        loss = ones.sub(GIoU)
        zeros = torch.zeros_like(loss)
        loss = torch.where(loss < 0, zeros, loss)
        return torch.sum(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
