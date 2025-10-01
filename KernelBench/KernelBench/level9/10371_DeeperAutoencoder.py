import torch
import torch.nn as nn
import torch.nn.functional as F


class DeeperAutoencoder(nn.Module):

    def __init__(self, input_length, output_length=None, neuron_multiplier=
        1, sigmoid=False, drop=False, drop_pct=0.3):
        """
        Dense deeper autoencoder. 
        
        Args:
            input_length (int): Length (i.e., size) of input sample.
            output_length (int): Length of output sample. Defaults to None, leave as default if 
                                 input and output are equal size.
            neuron_multiplier (int): Number to augment the set number of neurons. Defaults to 1.
            sigmoid (boolean): Defaults to False. Leave as default if output is nn.Linear (not sigmoid).
            drop (boolean): Defaults to False. True activates dropout for each layer except final.
            drop_pct (float): Amount of dropout to use (if drop is True). Defaults to 0.3.
            
        """
        super(DeeperAutoencoder, self).__init__()
        self.input_length = input_length
        if not output_length:
            self.output_length = input_length
        if output_length:
            self.output_length = output_length
        self.neuron_multiplier = neuron_multiplier
        self.sigmoid = sigmoid
        self.drop = drop
        self.drop_pct = drop_pct
        self.layer1 = nn.Linear(in_features=self.input_length, out_features
            =4 * 256 * self.neuron_multiplier)
        self.layer2 = nn.Linear(in_features=4 * 256 * self.
            neuron_multiplier, out_features=256 * self.neuron_multiplier)
        self.layer5 = nn.Linear(in_features=256 * self.neuron_multiplier,
            out_features=4 * 256 * self.neuron_multiplier)
        self.layer6 = nn.Linear(in_features=4 * 256 * self.
            neuron_multiplier, out_features=self.output_length)
        self.layer7 = nn.Linear(in_features=self.output_length,
            out_features=int(self.input_length / 3) + self.input_length)
        self.out = nn.Linear(in_features=int(self.input_length / 3) + self.
            input_length, out_features=self.output_length)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        if self.sigmoid:
            x = torch.sigmoid(self.out(x))
        if not self.sigmoid:
            x = self.out(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_length': 4}]
