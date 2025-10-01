import torch
import torch.utils.data


class Dnn_net_Loss(torch.nn.Module):

    def __init__(self):
        super(Dnn_net_Loss, self).__init__()

    def forward(self, model_output, targ_input):
        criterion = torch.nn.MSELoss(reduction='none')
        criterion
        targ_input = torch.cat((targ_input[:, :, 0], targ_input[:, :, 1]), 1)
        loss = criterion(model_output, targ_input)
        loss = torch.where(loss > 0, torch.sqrt(torch.tensor(2)) * loss, loss)
        mean_loss = torch.mean(loss)
        return mean_loss


def get_inputs():
    return [torch.rand([4, 4, 8, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
