import math
import torch


class AbstractTorchModule(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

    def save(self, path):
        None
        torch.save(self.state_dict(), path)

    def load(self, path):
        None
        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_device(self, device):
        self.device = device
        self


class GNNExplainerProbe(AbstractTorchModule):

    def __init__(self, num_edges, num_layers, init_strategy='normal',
        const_val=1.0):
        super(GNNExplainerProbe, self).__init__()
        mask = torch.empty((num_layers, num_edges))
        if init_strategy == 'normal':
            std = torch.nn.init.calculate_gain('relu') * math.sqrt(2.0 / (2 *
                math.sqrt(num_edges)))
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == 'const':
            torch.nn.init.constant_(mask, const_val)
        self.mask = torch.nn.Parameter(mask)

    def forward(self):
        s = torch.sigmoid(self.mask)
        mask_ent = -s * torch.log(s) - (1 - s) * torch.log(1 - s)
        penalty = mask_ent.mean() + 0.5 * s.sum()
        return s, penalty


def get_inputs():
    return []


def get_init_inputs():
    return [[], {'num_edges': 4, 'num_layers': 1}]
